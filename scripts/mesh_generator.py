import os
os.environ["OPEN3D_MUTE_LOG"] = "true"
import open3d as o3d
import numpy as np

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

class MeshGenerator:
    """
    create_box(width=1.0, height=1.0, depth=1.0, create_uv_map=False, map_texture_to_each_face=False)
    create_sphere(radius=1.0, resolution=20, create_uv_map=False)
    create_cylinder(radius=1.0, height=2.0, resolution=20, split=4, create_uv_map=False)
    create_cone(radius=1.0, height=2.0, resolution=20, split=1, create_uv_map=False)
    create_tetrahedron(radius=1.0, create_uv_map=False)
    create_icosahedron(radius=1.0, create_uv_map=False)
    create_octahedron(radius=1.0, create_uv_map=False)
    create_torus(torus_radius=1.0, tube_radius=0.5, radial_resolution=30, tubular_resolution=20)
    create_coordinate_frame(size=1.0, origin=array([0., 0., 0.]))
    create_arrow(cylinder_radius=1.0, cone_radius=1.5, cylinder_height=5.0, cone_height=4.0, resolution=20, cylinder_split=4, cone_split=1)
    """
    def __init__(self):
        self.low_limit = 0.2
        self.high_limit = 1.0

    def create_mesh(self, idx):
        if idx == 0: return self.create_box()
        elif idx == 1: return self.create_sphere()
        elif idx == 2: return self.create_cylinder()
        elif idx == 3: return self.create_cone()
        elif idx == 4: return self.create_tetrahedron()
        elif idx == 5: return self.create_icosahedron()
        elif idx == 6: return self.create_octahedron()
        elif idx == 7: return self.create_torus()
        else: self.exception(idx)

    def create_box(self):
        name = 'box'
        box_size = np.random.uniform(self.low_limit, self.high_limit, size=3)
        params = {'box_size':box_size}
        mesh = o3d.geometry.TriangleMesh.create_box(width=box_size[0], height=box_size[1], depth=box_size[2], create_uv_map=False, map_texture_to_each_face=False)
        return name, params, mesh

    def create_sphere(self):
        name = 'sphere'
        radius = np.random.uniform(self.low_limit, self.high_limit)
        params = {'radius':radius}
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20, create_uv_map=False)
        return name, params, mesh

    def create_cylinder(self):
        name = 'cylinder'
        radius = np.random.uniform(self.low_limit, self.high_limit)
        height = np.random.uniform(self.low_limit, self.high_limit)
        params = {'radius':radius, 'height':height}
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=20, split=4, create_uv_map=False)
        return name, params, mesh
    
    def create_cone(self):
        name = 'cone'
        radius = np.random.uniform(self.low_limit, self.high_limit)
        height = np.random.uniform(self.low_limit, self.high_limit)
        params = {'radius':radius, 'height':height}
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=20, split=1, create_uv_map=False)
        return name, params, mesh
    
    def create_tetrahedron(self):
        name = 'tetrahedron'
        radius = np.random.uniform(self.low_limit, self.high_limit)
        params = {'radius':radius}
        mesh = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius, create_uv_map=False)
        return name, params, mesh
    
    def create_icosahedron(self):
        name = 'icosahedron'
        radius = np.random.uniform(self.low_limit, self.high_limit)
        params = {'radius':radius}
        mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=radius/2, create_uv_map=False)
        return name, params, mesh
    
    def create_octahedron(self):
        name = 'octahedron'
        radius = np.random.uniform(self.low_limit, self.high_limit)
        params = {'radius':radius}
        mesh = o3d.geometry.TriangleMesh.create_octahedron(radius=radius, create_uv_map=False)
        return name, params, mesh
    
    def create_torus(self):
        name = 'torus'
        delta = 0.1
        torus_radius = np.random.uniform(self.low_limit, self.high_limit)
        tube_radius = np.random.uniform(torus_radius/2 - delta/2, torus_radius/2 + delta/2)
        params = {'torus_radius':torus_radius, 'tube_radius':tube_radius}
        mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=torus_radius, tube_radius=tube_radius, radial_resolution=30, tubular_resolution=20)
        return name, params, mesh
    
    def exception(self, idx):
        msg = (f"The id {idx} does not match any of the object ids:\n"
                    "box           :   0\n"
                    "sphere        :   1\n"
                    "cylinder      :   2\n"
                    "cone          :   3\n"
                    "tetrahedron   :   4\n"
                    "icosahedron   :   5\n"
                    "octahedron    :   6\n"
                    "torus         :   7\n")
        raise ValueError(msg)
