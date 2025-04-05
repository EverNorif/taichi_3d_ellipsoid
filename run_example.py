import numpy as np

from taichi_3d_ellipsoid.basic import EllipsoidRenderer, Ellipsoid

def random_generate_ellipsoids(num_ellipsoids: int, box_size: float = 10.0):
    centers, radii, colors = [], [], []
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

    return np.array(centers, dtype=np.float32), np.array(radii, dtype=np.float32), np.array(colors, dtype=np.float32)


if __name__ == "__main__":
    # generate random ellipsoids
    centers, radii, colors = random_generate_ellipsoids(num_ellipsoids=500, box_size=10.0)
    renderer = EllipsoidRenderer(centers, radii, colors, arr_type="numpy")
    renderer.run_gui()