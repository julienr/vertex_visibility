"""
This demonstrates how to determine vertex visibility by using the OpenGL depth
buffer.
Given a camera with parameters K (intrinsics), (R, C) (extrinsics), we want
to determine which vertices of a mesh are visible and which aren't (because
they are behind other faces from said mesh).

This is basically what opengl
does during depth testing. As far as I know, there is no way to get opengl to
tell us which vertex are visible (we could use occlusion queries). So we redo
the depth testing using OpenGL's depth buffer.

To do that, we render the mesh using OpenGL and get a copy of the depth buffer.
Then, for each vertex, we obtain the position it is projected to. We then
compare the depth stored in the depth buffer at that position with the distance
between the vertex and the camera center. If the depth buffer value is smaller
than the distance to the camera center, the vertex is occluded.
TODO: Display the virtual camera in the 3D views
"""
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
import norm
import pylab as pl

np.set_printoptions(precision=5, suppress=True)
##
verts, faces, _ = ply.load_ply('mesh2.ply')
verts = verts.astype(np.float32)
faces = np.array(faces, 'uint32')

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

# The camera for which we want to compute vertex visibility
K, R, C, imw, imh = get_camera_params()

# The depth map (just for visualization)
dimg = depth.compute_depth_map(verts, faces, K, R, C, imw, imh)

# Visible vertices computation from depth map
visible_verts, depths, dimg_depths = depth.compute_visible_verts(
    verts, faces, K, R, C, imw, imh, return_depth=True)
##
meshdata = vispy.geometry.MeshData(vertices=verts, faces=faces)

canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True,
        config={'depth_size': 24})

colormap = vispy.color.get_colormap('viridis')

grid = canvas.central_widget.add_grid()
grid.padding = 6

# will contain the depth map perceived by the virtual camera
vb1 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
# will contain the difference between depth map and depth measure from camera
# center
vb2 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)
# will contain visible vertices
vb3 = scene.widgets.ViewBox(border_color='white', parent=canvas.scene)

grid.add_widget(vb1, 0, 0)
grid.add_widget(vb2, 0, 1)
grid.add_widget(vb3, 0, 2)

#
depth_norm = norm.Normalize(#vmin=0,
                            vmin=min(depths.min(), dimg_depths.min()),
                            vmax=max(depths.max(), dimg_depths.max()))
# -------- vb1 : image
vimg = colormap.map(depth_norm(dimg)).reshape(dimg.shape[0], dimg.shape[1], -1)

image = scene.visuals.Image(vimg, interpolation='nearest')

vb1.add(image)
# Set 2D camera (the camera will scale to the contents in the scene)
vb1.camera = scene.PanZoomCamera(aspect=1)
# flip y-axis to have correct aligment
vb1.camera.flip = (0, 1, 0)
vb1.camera.set_range()
vb1.camera.zoom(1, (dimg.shape[1] / 2, dimg.shape[0] / 2))
vb1.bgcolor = '#efefef'

colorbar = scene.visuals.ColorBar(pos=[-20, 400], size=[400, 20],
        label='depth', cmap=colormap, orientation='left')
colorbar.clim = ('%.2f' % depth_norm.vmin, '%.2f' % depth_norm.vmax)
vb1.add(colorbar)

t = scene.visuals.Text('Depth map from OpenGL', parent=vb1, color='k')
t.font_size = 5
t.pos = (100, 20)

#
depth_diff = dimg_depths - depths
depth_diff_norm = norm.Normalize(vmin=depth_diff.min(),
                                 vmax=depth_diff.max())

# -------- vb2 : mesh with dimg_depth
vb2.camera = TurntableCamera(fov=45)
# The default depth_value of 100000 causes small artifacts. Use something
# smaller since our mesh uses meter units (so it has at most a height of ~2)
vb2.camera.depth_value = 10
vb2.camera.center = (0, 0, 0.7)
vb2.bgcolor = '#efefef'

mesh = scene.visuals.Mesh(meshdata=meshdata, shading='smooth', color='w')
# Need to enable cull_face
# https://github.com/vispy/vispy/issues/896#issuecomment-152083382
#mesh.set_gl_state('translucent', depth_test=True, cull_face=False)
vb2.add(mesh)

points = scene.visuals.Markers()
points.set_data(verts, edge_color=None,
                face_color=colormap.map(depth_norm(dimg_depths)), size=5)
points.set_gl_state('opaque', depth_test=True)
vb2.add(points)

axis = scene.visuals.XYZAxis()
vb2.add(axis)

t = scene.visuals.Text('Diff between vertex depth\nand depth map', parent=vb2,
                       color='k')
t.font_size = 5
t.pos = (120, 20)

# -------- vb3 : visible verts
vb3.camera = TurntableCamera(fov=45)
# The default depth_value of 100000 causes small artifacts. Use something
# smaller since our mesh uses meter units (so it has at most a height of ~2)
vb3.camera.depth_value = 10
vb3.camera.center = (0, 0, 0.7)
vb3.bgcolor = '#efefef'
vb3.camera.link(vb2.camera)

mesh = scene.visuals.Mesh(meshdata=meshdata, shading='smooth', color='w')
# Need to enable cull_face
# https://github.com/vispy/vispy/issues/896#issuecomment-152083382
#mesh.set_gl_state('translucent', depth_test=True, cull_face=False)
vb3.add(mesh)

points_vis = scene.visuals.Markers()
points_vis.set_data(verts[visible_verts], edge_color=None,
                    face_color=(0, 1, 0), size=5)
points_vis.set_gl_state('opaque', depth_test=True)
vb3.add(points_vis)
points_hid = scene.visuals.Markers()
points_hid.set_data(verts[~visible_verts], edge_color=None,
                    face_color=(1, 0, 0), size=5)
points_hid.set_gl_state('opaque', depth_test=True)
vb3.add(points_hid)

axis = scene.visuals.XYZAxis()
vb3.add(axis)

t = scene.visuals.Text('Visible vertices', parent=vb3,
                       color='k')
t.font_size = 5
t.pos = (120, 20)

##
if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()
##
