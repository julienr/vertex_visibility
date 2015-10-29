# A few notes about the PLY format supported by blender and meshlab
# Vertex indices in faces should be 'vertex_indices'
# Per-vertex texture coords should be 's', 't' (and not uv)
# Neither blender nor meshlab does seem to support per-face texcoords (using
# a texcoord list property with 6 entries)
#
# So we export with duplicate vertices, per-vertex texcoords. We use additional
# properties to map vertices to original SCAPE vertices (scape_vid)
import numpy as np
import collections

def load_ply(filename, load_bp=False):
    """
    Loads a .ply file.
    Returns verts, faces, faces_uv.
    If the .ply has no texture coordinates, faces_uv is an empty list

    if `load_bp` is True, will try to load property int32 body part and return
    an additional bpids array that gives the body part each vertex belongs too
    """
    with open(filename) as f:
        return load_ply_fileobj(f, load_bp)

def load_ply_fileobj(fileobj, load_bp=False):
    """Same as load_ply, but takes a file-like object"""
    def nextline():
        """Read next line, skip comments"""
        while True:
            line = fileobj.readline()
            assert line != '' # eof
            if not line.startswith('comment'):
                return line.strip()

    assert nextline() == 'ply'
    assert nextline() == 'format ascii 1.0'
    line = nextline()
    assert line.startswith('element vertex')
    nverts = int(line.split()[2])
    #print 'nverts : ', nverts
    assert nextline() == 'property float x'
    assert nextline() == 'property float y'
    assert nextline() == 'property float z'
    line = nextline()

    return_bp = False
    if load_bp:
        assert line == 'property int32 bpid'
        return_bp = True
        line = nextline()
    elif line == 'property int32 bpid':
        load_bp = True
        return_bp = False
        line = nextline()
    assert line.startswith('element face')
    nfaces = int(line.split()[2])
    #print 'nfaces : ', nfaces
    assert nextline() == 'property list uchar int vertex_indices'
    line = nextline()
    has_texcoords = line == 'property list uchar float texcoord'
    if has_texcoords:
        assert nextline() == 'end_header'
    else:
        assert line == 'end_header'

    # Verts
    if load_bp:
        bpids = np.zeros(nverts, dtype=int)
    verts = np.zeros((nverts, 3))
    for i in xrange(nverts):
        vals = nextline().split()
        verts[i,:] = [float(v) for v in vals[:3]]
        if load_bp:
            bpids[i] = int(vals[3])
    # Faces
    faces = []
    faces_uv = []
    for i in xrange(nfaces):
        vals = nextline().split()
        assert int(vals[0]) == 3
        faces.append([int(v) for v in vals[1:4]])
        if has_texcoords:
            assert len(vals) == 11
            assert int(vals[4]) == 6
            faces_uv.append([(float(vals[5]), float(vals[6])),
                             (float(vals[7]), float(vals[8])),
                             (float(vals[9]), float(vals[10]))])
            #faces_uv.append([float(v) for v in vals[5:]])
        else:
            assert len(vals) == 4
    if return_bp:
        return verts, faces, faces_uv, bpids
    else:
        return verts, faces, faces_uv
