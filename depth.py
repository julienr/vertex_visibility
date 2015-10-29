"""
Utility functions that uses OpenGL to draw meshes. This is *NOT* for
visualization, but for computating things like depth buffer, etc...

This was tested with vispy commit
9195c7ee1446bb7e08ea0c2e919b3d7cb2b46131

This requires pyopengl
"""
# Inspired by
# https://github.com/vispy/vispy/blob/master/examples/demo/gloo/offscreen.py
# https://github.com/vispy/vispy/blob/master/examples/basics/gloo/rotate_cube.py
import numpy as np
import numpy.ma as ma
from vispy import gloo
import vispy.gloo.gl as gl
from vispy import app
from vispy.util.ptime import time
from vispy.gloo.util import _screenshot
from vispy.util.transforms import perspective, translate, rotate, frustum
from volumit.volumit2d.scape2d import points_depth
from OpenGL import GL
import sklearn.neighbors as skneighbors
import cam

#app.use_app('pyglet')

# Getting the depth from the depth buffer in OpenGL is doable, see here :
#   http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
#   http://web.archive.org/web/20130426093607/http://www.songho.ca/opengl/gl_projectionmatrix.html
#   http://stackoverflow.com/a/6657284/116067
# But it is hard to get good precision, as explained in this article :
# http://dev.theomader.com/depth-precision/
#
# The thing is, we do not really want the depth buffer depth. We just want,
# for each fragment, its distance to the camera center. Once the vertex is
# in view space (view * model * v), this is simply the Z axis.
# So instead of reading from the depth buffer and undoing the projection
# matrix, we store the Z coord of each vertex in the COLOR buffer and then
# read from the color buffer. OpenGL desktop allows for float32 color buffer
# components.
DEPTH_VERTEX = """
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
attribute vec3 a_position;
//attribute vec3 a_normal;
attribute vec3 a_color;
varying vec4 v_color;
varying float view_depth;

void main()
{
    v_color = vec4(a_color, 1.0);
    vec4 view_pos = u_view * u_model * vec4(a_position, 1.0);
    // OpenGL Z axis goes out of the screen, so depths are negative. We reverse
    // the Z axis to point towards the target to get positive depths
    view_depth = -view_pos.z;
    gl_Position = u_projection * view_pos;
}
"""

DEPTH_FRAG = """
varying vec4 v_color;
varying float view_depth;
void main()
{
    //gl_FragColor = v_color;
    gl_FragColor = vec4(view_depth, view_depth, view_depth, 1.0);
}
"""


def gl_proj_matrix_from_K(width, height, K, near, far):
    # See https://github.com/googlesamples/tango-examples-c/blob/master/tango-gl/camera.cpp
    # https://developers.google.com/project-tango/overview/intrinsics-extrinsics
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xscale = near / float(fx)
    yscale = near / float(fy)
    xoffset = (cx - (width / 2.0)) * xscale
    yoffset = (cy - (height / 2.0)) * yscale

    # vispy return column-major to be OpenGL compatible
    return frustum(xscale * (-width  / 2.0) - xoffset,
                   xscale * ( width  / 2.0) - xoffset,
                   yscale * (-height / 2.0) - yoffset,
                   yscale * ( height / 2.0) - yoffset,
                   near,
                   far).T


def read_fbo_color_rgba32f(fbo):
    """
    Read the color attachment from a FBO, assuming it is GL_RGBA_32F
    """
    buffer = fbo.color_buffer
    h, w = buffer.shape[:2]
    x, y = 0, 0
    im = gl.glReadPixels(x, y, w, h, GL.GL_RGBA, GL.GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im.shape = h, w, 4
    im = im[::-1, :]
    return im


class DepthCanvas(app.Canvas):
    def __init__(self, verts, tris, K, R, C, size, verbose=False):
        # We hide the canvas upon creation.
        app.Canvas.__init__(self, show=False, size=size)
        self._t0 = time()
        # Create FBO to render the scene
        self._depthbuf = gloo.RenderBuffer(shape=size[::-1], format='depth')
        self._coltex = gloo.Texture2D(
                shape=(size[1], size[0], 4),
                format=GL.GL_RGBA,
                internalformat=GL.GL_RGBA32F)

        self._fbo = gloo.FrameBuffer(self._coltex, self._depthbuf)
        self.program = gloo.Program(DEPTH_VERTEX, DEPTH_FRAG)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])

        if True:
            # Autocompute near and far planes to maximize precision
            verts_cam = R.dot(verts[:, :3].T - C.reshape(-1, 1)).T
            depths = verts_cam[:,2]
            self.near = 10 ** np.floor(np.log10(depths.min()))
            self.far = 10 ** np.ceil(np.log10(depths.max()))
            if verbose:
                print "depths ", depths.min(), depths.max()
                print 'autonear|far : ', self.near, self.far
        else:
            self.near = 0.01
            self.far = 100.0

        # IMPORTANT: Opengl uses column-major matrices. Numpy uses row-major
        # by default. vispy works with transposed matrices so they can
        # directly be send to opengl. So the translate() from vispy acts like
        # this :
        #   In [81]: vispy.util.transforms.translate((1, 2, 3))
        #   Out[81]:
        #   array([[ 1.,  0.,  0.,  0.],
        #          [ 0.,  1.,  0.,  0.],
        #          [ 0.,  0.,  1.,  0.],
        #          [ 1.,  2.,  3.,  1.]])
        #
        self.projection = gl_proj_matrix_from_K(size[0], size[1], K,
                                                self.near, self.far)
        self.model = np.eye(4, dtype=np.float32)

        self.view = np.eye(4, dtype=np.float32)
        self.view[:3,:3] = R
        self.view[:3, 3] = -R.dot(C)

        # Our camera has z pointing towards the target while opengl has
        # z pointing outside of the screen => negate
        SHAPY_TO_OGL = np.array([
            [ 1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0,  1]], dtype=np.float32)
        self.view = SHAPY_TO_OGL.dot(self.view)

        # transpose to make them opengl row-major compatible (that's what
        # vispy does)
        self.program['u_projection'] = self.projection.T.copy()
        self.program['u_model'] = self.model.T.copy()
        self.program['u_view'] = self.view.T.copy()
        if verbose:
            print 'model\n', self.model
            print 'proj\n', self.projection
            print 'view matrix\n', self.view

        gloo.set_clear_color('black')
        gloo.set_state('opaque')

        # Create
        nverts = verts.shape[0]
        vtype = [('a_position', np.float32, 3),
                 ('a_color', np.float32, 3)]
        verts_colors = np.ones((nverts, 3), dtype=np.float32)
        # TODO: Can apply a colormap to body parts to show them
        #verts_colors = verts[:,3].astype(np.float32)
        # V should contain [(v[0], c[0]), (v[1], c[1])] where v and c are
        # 3-vector containing the position, respectively color of each vertex
        V = [(verts[i, :3], verts_colors[i]) for i in xrange(nverts)]
        V = np.array(V, dtype=vtype)
        I = np.array(tris, dtype=np.uint16).ravel()
        self.tris_buf = gloo.IndexBuffer(I)
        self.verts_buf = gloo.VertexBuffer(V)

        # Manually force update
        self.update()

    def on_draw(self, event):
        with self._fbo:
            gloo.clear('black')
            self.program.bind(self.verts_buf)
            self.program.draw('triangles', self.tris_buf)
            # Retrieve depth
            # For some reason, vispy flipuds images in read_pixels, so we
            # unflip here. This is inefficient but vispy-compatible
            self.depth = read_fbo_color_rgba32f(self._fbo)
            self.depth = np.flipud(self.depth)
        self._time = time() - self._t0
        app.quit()


def compute_depth_map(verts, tris, K, R, C, imwidth, imheight, verbose=False):
    """
    Renders the given mesh to texture, returning the color and depth buffers
    """
    c = DepthCanvas(verts, tris, K, R, C, size=(imwidth, imheight),
                    verbose=verbose)
    size = c.size
    # This blocks until render is finished, which is what we wants
    c.render()

    # The rendering is done, we get the rendering output (4D NumPy array)
    if verbose:
        print('Finished in %.1fms.' % (c._time*1e3))
    depth = ma.masked_where(c.depth[:,:,0] < 1e-6, c.depth[:,:,0])
    return depth


def get_depth_for_points(p2d, depth_map, dist_thresh_px=3):
    """
    This is a depth map lookup function. We have a list of 2D points and want
    to get their depth as computed by the depth map.
    The obvious solution would be depth_map[p2d[:,0], p2d[:,1]], but due
    to rounding, some edge vertices will fall outside of the rendered mesh and
    have a depth of 'far'. So instead, we do a NN search on the depth
    buffer pixels with a valid depth and assign the depth of the nearest depth
    buffer pixel, with a maximum distance to avoid assigning depth to point
    really outside of the rendered mesh.

    Returns:
        - depths: An array containing the depth of each point in p2d. Can
                  contain nan for invalid points
    """
    depth_nn = skneighbors.NearestNeighbors(n_neighbors=1)
    ij = np.transpose(np.nonzero(~depth_map.mask))
    ij_depth = depth_map[~depth_map.mask].filled(-1)
    depth_nn.fit(ij[:,::-1])

    dist, ind = depth_nn.kneighbors(p2d)
    dist = np.squeeze(dist)
    ind = np.squeeze(ind)

    p2d_depths = np.zeros(p2d.shape[0], dtype=np.float)
    p2d_depths[:] = np.nan

    valids = dist < dist_thresh_px
    p2d_depths[valids] = ij_depth[ind[valids]]
    return p2d_depths


def view_depth(points, R, C):
    X = R.dot(points[:, :3].T - C.reshape(-1, 1)).T
    depths = X[:,2]
    return depths


def compute_visible_verts(verts, faces, K, R, C, imw, imh,
                          bias=0.01, return_depth=False):
    """
    Given a mesh and a camera, returns a mask indicating which vertices are
    visible. This uses OpenGL to render the mesh.
    """
    dimg = compute_depth_map(verts, faces, K, R, C, imw, imh)
    p2d = cam.project_on_camera(verts, K, R, C).T
    dimg_depths = get_depth_for_points(p2d, dimg)

    depths = view_depth(verts, R, C)
    visible_verts = depths <= dimg_depths + bias
    if return_depth:
        return visible_verts, depths, dimg_depths
    else:
        return visible_verts

