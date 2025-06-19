import numpy as np
import trimesh

def load_model(path):
    loaded = trimesh.load_mesh(path, process=False)

    # If the loaded object is a Scene, extract first mesh
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError("No geometry found in scene")
        mesh = next(iter(loaded.geometry.values()))
    else:
        mesh = loaded # It's already a Trimesh!
        
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32).flatten()

    # UVs: trimesh stores them in mesh.visual.uv if available
    uvs = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = np.array(mesh.visual.uv, dtype=np.float32)
        if uvs.shape[1] > 2:
            uvs = uvs[:, :2]

    # Normals: trimesh can have vertex normals or face normals
    normals = None
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
        normals = np.array(mesh.vertex_normals, dtype=np.float32)

    return {
        'vertices': vertices,
        'faces': faces,
        'uvs': uvs,
        'normals': normals,
    }