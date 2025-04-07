import taichi as ti
from typing import Tuple
from .basic import EllipsoidRenderer

# Render Ellipsoids using rasterization
# Reference: https://blog.42yeah.is/rendering/opengl/2023/12/20/rasterizing-splats.html
class EllipsoidRasterizationRenderer(EllipsoidRenderer):
    def __init__(
        self,
        centers,
        radii,
        colors,
        rotations,
        opacities,
        arr_type: str="numpy",  # numpy or torch
        res:Tuple[int, int]=(1024, 1024),
        camera_pos:Tuple[float, float, float]=(0.0, 0.0, 5.0),
        camera_lookat:Tuple[float, float, float]=(0.0, 0.0, 0.0),
        camera_up:Tuple[float, float, float]=(0.0, 1.0, 0.0),
        fov:float=60.0,
        background_color:Tuple[float, float, float]=(0.05, 0.05, 0.05),
        ambient:float=0.2,
        diffuse_strength:float=0.6,
        shininess:float=32.0,
        specular_strength:float=0.5,
        headless:bool=False,
        opacity_limit:float=0.2,
        ti_arch=ti.gpu,
        device:str="cuda:0",
        ):
        # initialize the base renderer
        super().__init__(
            centers=centers,
            radii=radii,
            colors=colors,
            rotations=rotations,
            opacities=opacities,
            arr_type=arr_type,
            res=res,
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_up=camera_up,
            fov=fov,
            background_color=background_color,
            ambient=ambient,
            diffuse_strength=diffuse_strength,
            shininess=shininess,
            specular_strength=specular_strength,
            headless=headless,
            opacity_limit=opacity_limit,
            ti_arch=ti_arch,
            device=device,
        )

        self.near_plane = 0.1
        self.far_plane = 100.0
        self.depth_buffer = ti.field(dtype=ti.f32, shape=(self.res_x, self.res_y))

    @ti.kernel
    def render(self):
        camera_pos = self.cam_pos[None]
        camera_lookat = self.cam_lookat[None]
        camera_up = self.cam_up[None]
        fov_radians = self.start_fov * ti.math.pi / 180.0
        
        # calculate the camera coordinate system
        forward = (camera_lookat - camera_pos).normalized()
        right = ti.math.cross(forward, camera_up).normalized()
        up = ti.math.cross(right, forward).normalized()
        
        # field of view parameters
        half_height = ti.tan(fov_radians / 2.0)
        half_width = half_height * self.res_x / self.res_y
        viewport_height = 2.0 * half_height
        viewport_width = 2.0 * half_width

        # clear the buffer
        for i, j in self.pixels:
            self.pixels[i, j] = self.background_color
            self.depth_buffer[i, j] = 1e6
        
        # preprocess frustum culling
        for e_idx in range(self.num_ellipsoids):
            if self.ellipsoids[e_idx].opacity >= self.opacity_limit:
                if self.is_ellipsoid_visible(
                    camera_pos, forward, right, up, half_width, half_height,
                    self.ellipsoids[e_idx].center, self.ellipsoids[e_idx].radii
                ):
                    self.visible[e_idx] = 1
                else:
                    self.visible[e_idx] = 0
            else:
                self.visible[e_idx] = 0
        
        # render each ellipsoid
        for k in range(self.num_ellipsoids):
            # calculate the distance to the camera
            to_center = self.ellipsoids[k].center - camera_pos
            depth = to_center.dot(forward)
            
            if depth > 0 and self.visible[k] == 1:
                # calculate the screen space position
                screen_x = to_center.dot(right) / depth * self.res_y / viewport_height + self.res_x / 2
                screen_y = to_center.dot(up) / depth * self.res_y / viewport_height + self.res_y / 2
                
                # calculate the screen space radius, considering the maximum size of the ellipsoid
                max_scale = ti.max(self.ellipsoids[k].radii[0], ti.max(self.ellipsoids[k].radii[1], self.ellipsoids[k].radii[2]))
                screen_radius = ti.sqrt(
                    (max_scale * right).norm() ** 2 + 
                    (max_scale * up).norm() ** 2
                ) / depth * self.res_y / viewport_height
                
                # calculate the bounding box range
                min_x = ti.max(0, ti.i32(screen_x - screen_radius - 2))
                min_y = ti.max(0, ti.i32(screen_y - screen_radius - 2))
                max_x = ti.min(self.res_x, ti.i32(screen_x + screen_radius + 2))
                max_y = ti.min(self.res_y, ti.i32(screen_y + screen_radius + 2))
                
                # only iterate over the pixels in the bounding box
                for i, j in ti.ndrange((min_x, max_x), (min_y, max_y)):
                    # calculate the ray direction
                    u = (i - self.res_x / 2) * viewport_width / self.res_x
                    v = (j - self.res_y / 2) * viewport_height / self.res_y
                    ray_dir = (forward + u * right + v * up).normalized()
                    
                    hit, t, normal = self.ray_ellipsoid_intersection(
                        camera_pos, ray_dir, self.ellipsoids[k].center, self.ellipsoids[k].radii, self.ellipsoids[k].rotation)
                    
                    if hit:
                        view_space_depth = (camera_pos + ray_dir * t - camera_pos).dot(forward)
                        pixel_depth = (view_space_depth - self.near_plane) / (self.far_plane - self.near_plane)
                        
                        old_depth = ti.atomic_min(self.depth_buffer[i, j], pixel_depth)
                        if pixel_depth <= old_depth:
                            light_to_point = (camera_pos + ray_dir * t - self.light_position[None]).normalized()
                            
                            # ambient light
                            ambient_component = self.ambient * self.ellipsoids[k].color
                            # diffuse
                            diffuse = max(normal.dot(-light_to_point), 0.0)
                            diffuse_component = self.diffuse_strength * diffuse * self.ellipsoids[k].color
                            # specular
                            half_vec = (-light_to_point + (-ray_dir)).normalized()
                            specular = pow(max(normal.dot(half_vec), 0.0), self.shininess)
                            specular_component = self.specular_strength * specular * ti.Vector([1.0, 1.0, 1.0])
                            
                            self.pixels[i, j] = ambient_component + diffuse_component + specular_component
        