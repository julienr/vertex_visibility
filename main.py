##
import matplotlib
matplotlib.use('Qt4Agg')
import sys
import numpy as np
import numpy.linalg as la
from vispy import scene
from vispy.color import Color
from vispy import gloo
from vispy.scene.cameras import TurntableCamera
import vispy.io
import vispy.geometry
import ply
import depth
from norm import Normalize
import pylab as pl

np.set_printoptions(precision=5, suppress=True)
##
verts, faces, _ = ply.load_ply('mesh2.ply')
verts = verts.astype(np.float32)
faces = np.array(faces, 'uint32')
meshdata = vispy.geometry.MeshData(vertices=verts, faces=faces)

canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True,
        config={'depth_size': 24})

# Set up a viewbox to display the cube with interactive arcball
view = canvas.central_widget.add_view()
#view.bgcolor = '#efefef'
view.bgcolor = '#efefef'
view.camera = TurntableCamera(fov=45)
# The default depth_value of 100000 causes small artifacts. Use something
# smaller since our mesh uses meter units (so it has at most a height of ~2)
view.camera.depth_value = 10
view.camera.center = (0, 0, 0.7)
print view.camera

mesh = scene.visuals.Mesh(meshdata=meshdata, shading='smooth', color='w')
# Need to enable cull_face
# https://github.com/vispy/vispy/issues/896#issuecomment-152083382
#mesh.set_gl_state('translucent', depth_test=True, cull_face=False)
view.add(mesh)


# Add a 3D axis to keep us oriented
axis = scene.visuals.XYZAxis(parent=view.scene)

def rotx(angle):
    """Rotation matrix of angle (in radians) around the x-axis"""
    cosa = np.cos(angle)
    sina = np.sin(angle)
    return np.array([[1, 0, 0], [0, cosa, -sina], [0, sina, cosa]], dtype=float)

def roty(angle):
    """Rotation matrix of angle around the y-axis"""
    cosa = np.cos(angle)
    sina = np.sin(angle)
    return np.array([[cosa, 0, sina], [0, 1, 0], [-sina, 0, cosa]], dtype=float)

def get_camera_params():
    K = np.array([
        [ 625.,    0.,  240.],
        [   0.,  625.,  320.],
        [   0.,    0.,    1.]], dtype=np.float32)
    imw = 450
    imh = 800

    R = np.array([
        [-0.023855,  0.336322, -0.941444],
        [-0.999549, -0.025208,  0.016322],
        [-0.018243,  0.941409,  0.336772]], dtype=np.float32)
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]], dtype=np.float32)
    assert np.allclose(la.det(R), 1, atol=1e-4)
    R = rotx(0.5).dot(roty(-0.1).dot(R))

    C = np.array([0, -3, 3], dtype=np.float32)

    return K, R, C, imw, imh

K, R, C, imw, imh = get_camera_params()

if False:
    # This shows the depth map
    dimg = depth.compute_depth_map(verts, faces, K, R, C, imw, imh)
    pl.figure()
    pl.title('depth map')
    pl.imshow(dimg)
    pl.colorbar()
    pl.show()
##
reload(depth)
visible_verts, depths, dimg_depths = depth.compute_visible_verts(
    verts, faces, K, R, C, imw, imh, return_depth=True)

colormap = vispy.color.get_colormap('autumn')
#vmin = min(depths.min(), dimg_depths.min())
#vmax = max(depths.max(), dimg_depths.max())

depth_norm = Normalize()
dimg_norm = Normalize()


points = scene.visuals.Markers()
#points.set_data(verts, edge_color=None, face_color=(1, 0, 0, .5), size=5)
points.set_data(verts, edge_color=None,
                face_color=colormap.map(dimg_norm(depths)), size=5)
view.add(points)

colorbar = scene.visuals.ColorBar(pos=[100, 300], size=[400, 20],
        label='depth', cmap=colormap, orientation='left')
colorbar.clim = ('%.2f' % dimg_norm.vmin, '%.2f' % dimg_norm.vmax)

# Add the 2d view first so it is behind the 3d view (for interaction)
view_2d = canvas.central_widget.add_view()
view_2d.bgcolor = (0, 0, 0, 0) # transparent
view_2d.interactive = False
view_2d.add(colorbar)

##

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()
##
