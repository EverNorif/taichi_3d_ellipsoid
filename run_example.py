import numpy as np

from taichi_3d_ellipsoid.basic import EllipsoidRenderer, Ellipsoid

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


if __name__ == "__main__":
    example_case = 0 # PARAMETERS HERE

    if example_case == 0:
        # generate random ellipsoids
        print("Example 1: random ellipsoids from numpy")
        centers, radii, colors, rotations, opacities = random_generate_ellipsoids(
            num_ellipsoids=500, # PARAMETERS HERE
            box_size=10.0 # PARAMETERS HERE
        )
        renderer = EllipsoidRenderer(
            centers=centers,
            radii=radii,
            colors=colors,
            rotations=rotations,
            opacities=opacities,
            arr_type="numpy"
        )
        renderer.run_gui()
    else:
        raise ValueError(f"Unsupported case: {example_case}")