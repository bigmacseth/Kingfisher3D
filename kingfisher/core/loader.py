import numpy as np
from pyassimp import load, release

def load_model(path):
    with load(path) as scene:
        assert len(scene.meshes)
        mesh = scene.meshes[0]

        assert len(mesh.vertices)
        print(mesh.vertices[0])
        
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32).flatten()

    # Optional: UVs and normals if I want to extend this later

    uvs = None
    if mesh.texturecoords is not None and len(mesh.texturecoords) > 0 and mesh.texturecoords[0].shape[1] >= 2:
        uvs = np.array(mesh.texturecoords[0], dtype=np.float32)[:, :2]
    else:
        uvs = None

    if mesh.normals is not None and mesh.normals.shape[1] == 3:
        normals = np.array(mesh.normals, dtype=np.float32)

    return {
        'vertices': vertices,
        'faces': faces,
        'uvs': uvs,
        'normals': normals,
    }