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

# TODO: Deprecated
def load_ply_fileobj(fileobj, load_bp=False):
    """Same as load_ply, but takes a file-like object"""
    def nextline():
        """Read next line, skip comments"""
        while True:
            line = fileobj.readline()
            assert line != '' # eof
            if not line.startswith('comment'):
                return line.strip()

    print 'load_ply_fileobj is deprecated'

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


class PLYFile(object):
    """Helper class to load .ply files"""
    def __init__(self, fileobj):
        self.f = fileobj
        # We collect comments because they can contain metadata
        self.comments = []

    def nextline(self):
        """Read next line, skip comments"""
        while True:
            line = self.f.readline()
            assert line != '' # eof
            if line.startswith('comment'):
                self.line = line.strip()
                self.comments.append(self.line)
            else:
                self.line = line.strip()
                return self.line


# Map PLY types to python types
PROP_TYPE_MAP = {
    'int' : 'int',
    'int32' : 'int',
    'float' : 'float',
    'float32' : 'float',
}


def _parse_header(ply_file):
    """Parses a PLY header"""
    assert ply_file.nextline() == 'ply'
    assert ply_file.nextline() == 'format ascii 1.0'
    ply_file.nextline()

    # Build a dict of the various elements and their properties.
    # For example
    #   element vertex 500
    #   property float x
    #   property list uchar int mylist
    # Will yield
    # elements['vertex'] = {'x' : (float,), 'mylist' : ('list', uchar, int)}
    elements = collections.OrderedDict()

    while ply_file.line != 'end_header':
        assert ply_file.line.startswith('element'), ply_file.line
        elname, count = ply_file.line.split()[1:]
        elements[elname] = collections.OrderedDict()
        elements[elname]['_count'] = int(count)

        while ply_file.nextline().startswith('property'):
            # A property is either 'property <type> <name>'
            # or 'property list <lentype> <elmtype> <name>' for a list
            v = ply_file.line.split()
            if len(v) == 3:
                prop_type, prop_name = v[1], v[2]
                elements[elname][prop_name] = (PROP_TYPE_MAP[prop_type],)
            else:
                assert len(v) == 5
                assert v[1] == 'list'
                assert v[2] == 'uchar', 'list len should be uchar'
                elm_type, prop_name = v[3], v[4]
                elements[elname][prop_name] = ('list', PROP_TYPE_MAP[elm_type])

    # Look into the comment for info about texture filename
    metadata = {}
    for c in ply_file.comments:
        if c.startswith('comment TextureFile'):
            v = c.split()
            if len(v) > 2:
                metadata['texturefile'] = v[2]

    return elements, metadata


def _parse_property(propspec, line_entries):
    """
    Parse a property from a list of space-separated entries in a line
    Returns the number of entries used
    """
    if propspec[0] == 'list':
        lstlen = int(line_entries[0])
        entries = line_entries[1:1+lstlen]
        if propspec[1] == 'float':
            lst = [float(e) for e in entries]
        elif propspec[1] == 'int':
            lst = [int(e) for e in entries]
        else:
            assert False, propspec
        return lst, 1+lstlen
    else:
        if propspec[0] == 'float':
            return float(line_entries[0]), 1
        elif propspec[0] == 'int':
            return int(line_entries[0]), 1
        else:
            assert False, propspec


def load_ply_fileobj_to_rawdict(fileobj):
    """
    Load a PLY file to a dictionary. A simple PLY file will be turned into
        data['vertex'] = {'x': [0, 0.5], 'y': [0.2, 0.3], 'z' : [0.4, 0.5]}
        data['face'] = {'vertex_indices' : [[0, 1, 2]]}
    """
    plyfile = PLYFile(fileobj)
    header, metadata = _parse_header(plyfile)

    data = {}
    for elmname, propspecs in header.items():
        count = propspecs['_count']
        propdict = collections.defaultdict(lambda : [])
        for i in xrange(count):
            line = plyfile.nextline()
            line = line.split()
            lineidx = 0
            for propname, spec in propspecs.items():
                if propname == '_count':
                    continue
                value, nused = _parse_property(spec, line[lineidx:])
                lineidx += nused
                propdict[propname].append(value)
        data[elmname] = propdict

    # Merge metadata in our dict
    for k, v in metadata.items():
        assert k not in data
        data[k] = v
    return data


def load_ply_to_dict(filename):
    return load_ply_fileobj_to_dict(open(filename))


def load_ply_fileobj_to_dict(fileobj):
    """
    Load a PLY file to a dictionary.
    The dictionary will contain :
    - verts, faces
    - verts_uv, faces_uv, bpid, scape_vid (optional)
    """
    data = load_ply_fileobj_to_rawdict(fileobj)

    verts = np.c_[data['vertex']['x'],
                  data['vertex']['y'],
                  data['vertex']['z']]
    faces = data['face']['vertex_indices']
    data['verts'] = verts
    del data['vertex']['x'], data['vertex']['y'], data['vertex']['z']
    data['faces'] = faces
    del data['face']['vertex_indices']

    if 'u' in data['vertex']:
        verts_uv = np.c_[data['vertex']['u'], data['vertex']['v']]
        data['verts_uv'] = verts_uv
        del data['vertex']['u'], data['vertex']['v']

    if 'texcoord' in data['face']:
        faces_uv = []
        for entries in data['face']['texcoord']:
            faces_uv.append((entries[0], entries[1]), (entries[2], entries[3]),
                            (entries[4], entries[5]))
        data['faces_uv'] = faces_uv
        del data['face']['texcoord']

    if 'bpid' in data['vertex']:
        data['bpid'] = data['vertex']['bpid']
        del data['vertex']['bpid']

    if 'scape_vid' in data['vertex']:
        data['scape_vid'] = data['vertex']['scape_vid']
        del data['vertex']['scape_vid']

    return data


def save_ply(filename, verts, faces, verts_uv=None, faces_uv=None,
             scape_vids=None, bpids=None, texturefile=None):
    """
    Saves to a ply file.
    Note that you can specify only one of verts_uv or faces_uv
    Args:
        verts: A Nx3 index of vertices. If save_bp, should be Nx4 with the
               last column containing body part index
        faces: A list of 3-uple
        verts_uv: per-vertex uv (a Nx2 array)
        faces_uv: per-face uv (For each face, a list of 3 uv pairs)
        sacep_vids: For each vertex, the corresponding SCAPE vertex id
        bpids: For each vertex, the body part id
        texturefile: Optionaly specify the name of the texture file (this is
                     included as a comment in the file, but is understood by
                     meshlab)
    """
    assert not (verts_uv is not None and faces_uv is not None), \
        "Only one of verts_uv or faces_uv can be specified"

    # This is unsupported by blender, meshlab and the android viewer, so forbid
    # it. Use verts_uv instead
    assert not faces_uv, "Exporting with faces_uv is a bad idea for now"

    with open(filename, 'w') as f:
        # -- header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        if texturefile is not None:
            f.write('comment TextureFile %s\n' % texturefile)
        f.write('element vertex %d\n' % verts.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if verts_uv is not None:
            # Blender ply importer expects s,t
            f.write('property float s\n')
            f.write('property float t\n')
        if bpids is not None:
            f.write('property int32 bpid\n')
        if scape_vids is not None:
            f.write('property int32 scape_vid\n')
        f.write('element face %d\n' % len(faces))
        # Use vertex_indices and not vertex_index because Blender only
        # understands vertex_indices
        f.write('property list uchar int vertex_indices\n')
        if faces_uv is not None:
            f.write('property list uchar float texcoord\n')
        f.write('end_header\n')
        # -- verts
        for i in xrange(verts.shape[0]):
            f.write('%f %f %f' % tuple(verts[i, :3]))
            if verts_uv is not None:
                f.write(' %f %f' % tuple(verts_uv[i]))
            if bpids is not None:
                f.write(' %d' % bpids[i])
            if scape_vids is not None:
                f.write(' %d' % scape_vids[i])
            f.write('\n')
        # -- faces
        for i in xrange(len(faces)):
            f.write('3 %d %d %d' % (faces[i][0], faces[i][1], faces[i][2]))
            if faces_uv is not None:
                uvs = faces_uv[i][0] + faces_uv[i][1] + faces_uv[i][2]
                f.write(' 6 %f %f %f %f %f %f' % uvs)
            f.write('\n')
