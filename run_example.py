import numpy as np
import taichi as ti

from taichi_3d_ellipsoid import EllipsoidRayTracingRenderer

def euler_to_rotation_matrix(euler_angles):
    rx, ry, rz = euler_angles[0], euler_angles[1], euler_angles[2]
    
    # calculate the rotation matrix for each axis
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)
    
    R_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    
    R_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    
    R_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    
    R = R_z @ R_y @ R_x

    return R

def random_generate_ellipsoids(num_ellipsoids: int, box_size: float = 10.0):
    centers, radii, colors, rotations, opacities = [], [], [], [], []
    for i in range(num_ellipsoids):
        x = np.random.random() * box_size - box_size / 2.0  
        y = np.random.random() * box_size - box_size / 2.0
        z = np.random.random() * box_size - box_size / 2.0
        centers.append([x, y, z])
        
        base_radius = np.random.random()
        radii.append([
            base_radius,
            base_radius * (0.1 + np.random.random() * 0.8),
            base_radius * (0.1 + np.random.random() * 0.8)
        ])
        
        r, g, b = np.random.random(), np.random.random(), np.random.random()
        colors.append([r, g, b])

        euler_angles = np.random.random(3) * 2 * np.pi
        rotations.append(euler_to_rotation_matrix(euler_angles))

        opacities.append(np.random.random())

    return np.array(centers, dtype=np.float32), np.array(radii, dtype=np.float32), \
        np.array(colors, dtype=np.float32), np.array(rotations, dtype=np.float32), \
        np.array(opacities, dtype=np.float32)

def normalize(quaternions):
    norm = np.sqrt(np.sum(quaternions * quaternions, axis=-1, keepdims=True))
    norm = np.maximum(norm, 1e-8)
    return quaternions / norm

def sigmoid(x):
    x = np.clip(x, -88, 88)
    return 1.0 / (1.0 + np.exp(-x))

def quat_to_rotmat(quats):
    N = quats.shape[0]
    rotmats = np.zeros((N, 3, 3), dtype=np.float32)
    
    w = quats[:, 0]
    x = quats[:, 1]
    y = quats[:, 2]
    z = quats[:, 3]
    
    rotmats[:, 0, 0] = 1 - 2*y*y - 2*z*z
    rotmats[:, 0, 1] = 2*x*y - 2*w*z
    rotmats[:, 0, 2] = 2*x*z + 2*w*y
    
    rotmats[:, 1, 0] = 2*x*y + 2*w*z
    rotmats[:, 1, 1] = 1 - 2*x*x - 2*z*z
    rotmats[:, 1, 2] = 2*y*z - 2*w*x
    
    rotmats[:, 2, 0] = 2*x*z - 2*w*y
    rotmats[:, 2, 1] = 2*y*z + 2*w*x
    rotmats[:, 2, 2] = 1 - 2*x*x - 2*y*y
    
    return rotmats

def load_ellipsoids_from_ply(ply_file_path: str):
    from plyfile import PlyData

    ply_data = PlyData.read(ply_file_path)
    vertices = ply_data['vertex']
    num_points = len(vertices)

    centers = np.zeros((num_points, 3), dtype=np.float32)
    centers[:, 0] = vertices['x']
    centers[:, 1] = vertices['y']
    centers[:, 2] = vertices['z']

    radii = np.zeros((num_points, 3), dtype=np.float32)
    radii[:, 0] = vertices['scale_0']
    radii[:, 1] = vertices['scale_1']
    radii[:, 2] = vertices['scale_2']
    radii = np.exp(radii)

    colors = np.zeros((num_points, 3), dtype=np.float32)
    colors[:, 0] = vertices['f_dc_0']
    colors[:, 1] = vertices['f_dc_1']
    colors[:, 2] = vertices['f_dc_2']
    C0 = 0.28209479177387814
    colors = colors * C0 + 0.5

    quats = np.zeros((num_points, 4), dtype=np.float32)
    quats[:, 0] = vertices['rot_0']
    quats[:, 1] = vertices['rot_1']
    quats[:, 2] = vertices['rot_2']
    quats[:, 3] = vertices['rot_3']
    quats = normalize(quats) # wxyz
    rotations = quat_to_rotmat(quats)

    opacities = np.ones(num_points, dtype=np.float32)
    opacities[:] = vertices['opacity']
    opacities = sigmoid(opacities)

    return centers, radii, colors, rotations, opacities

if __name__ == "__main__":
    example_case = 1 # PARAMETERS HERE

    if example_case == 1:
        # generate random ellipsoids
        print("Example 1: random ellipsoids from numpy with ray tracing")
        centers, radii, colors, rotations, opacities = random_generate_ellipsoids(
            num_ellipsoids=500, # PARAMETERS HERE
            box_size=10.0 # PARAMETERS HERE
        )
        renderer = EllipsoidRayTracingRenderer(
            centers=centers,
            radii=radii,
            colors=colors,
            rotations=rotations,
            opacities=opacities,
            arr_type="numpy"
        )
        renderer.run_gui()
    elif example_case == 2:
        print("Example 2: load ellipsoids from 3dgs ply file and render with ray tracing")
        centers, radii, colors, rotations, opacities = load_ellipsoids_from_ply(
            ply_file_path="example.ply" # PARAMETERS HERE
        )
        renderer = EllipsoidRayTracingRenderer(
            centers=centers,
            radii=radii,
            colors=colors,
            rotations=rotations,
            opacities=opacities,
            arr_type="numpy",
            opacity_limit=0.2,
            background_color=(0.5, 0.5, 0.5),
            headless=True,
            ti_arch=ti.gpu,  # NOTE: if you use Mac, please change to ti.vulkan, otherwise the image can be incomplete
        )
        renderer.render_image(
            output_path="example.png",
            camera_pos=(0.0, -0.6, 0.0),
            camera_lookat=(0, 0, 0),
            camera_up=(0, 0, 1),
        )
    else:
        raise ValueError(f"Unsupported case: {example_case}")