# to read and write obj(w. mtl) file
import trimesh
import numpy as np
from collections import OrderedDict
import itertools

class materialed_meshes:
    """
    If the obj have mulitple meshes and materials, it will build a scene.
    And seperate meshes by material.

    Dump all meshes and concatenate them is a good way to accelerate ray-mesh intersection.
    But we need to record each vertex belongs which material. 
    """
    def __init__(self, obj_path):
        scene = trimesh.load(obj_path, force='scene')
        self.scene_dump = scene.dump()
        self.mesh_all = scene.dump(concatenate=True)

        self.vertex_cnt = [mesh.vertices.shape[0] for mesh in self.scene_dump]
        self.face_cnt = [mesh.faces.shape[0] for mesh in self.scene_dump]

        self.vertex_accu = list(itertools.accumulate(self.vertex_cnt))
        self.face_accu = list(itertools.accumulate(self.face_cnt))
    
    def find_material_and_idx(self, accu_list, idx):
        material_idx = np.searchsorted(accu_list, idx, side='right')
        local_idx = idx - accu_list[material_idx]
        return material_idx, local_idx

    def get_material_from_face_idx_of_all(self, face_idx_in_all):
        """
            face_idx_in_all:
                face idx in self.mesh_all.faces
        """
        material_idx, local_idx = self.find_material_and_idx(self.face_accu, face_idx_in_all)
        mesh = self.scene_dump[material_idx]
        face_local = mesh.faces[local_idx] # TrackedArray. local
        uv_local = mesh.visual.uv[face_local]
        material = mesh.visual.material
        material.kwargs['name'] = mesh.metadata['name']

        return material, face_local, uv_local
        

    def get_all_meshes(self):
        return self.mesh_all