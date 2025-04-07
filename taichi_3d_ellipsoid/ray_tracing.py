import taichi as ti
from typing import Tuple
from .basic import EllipsoidRenderer

# Render Ellipsoids using ray tracing
class EllipsoidRayTracingRenderer(EllipsoidRenderer):
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
        
        # iterate over each pixel
        for i, j in self.pixels:
            # calculate the normalized pixel coordinates (from -1 to 1)
            u = (i + 0.5) / self.res_x * 2 - 1
            v = (j + 0.5) / self.res_y * 2 - 1
            
            # calculate the ray direction
            ray_dir = (
                forward + 
                u * half_width * right + 
                v * half_height * up
            ).normalized()
            
            # default use the background color
            color = self.background_color
            
            # for tracking the nearest intersection
            closest_hit = False
            closest_t = 1e10
            closest_normal = ti.Vector([0.0, 0.0, 0.0])
            closest_color = ti.Vector([0.0, 0.0, 0.0])
            closest_idx = -1
            
            # detect the intersection between ray and all ellipsoids
            for e_idx in range(self.num_ellipsoids):
                if self.visible[e_idx] == 1:
                    hit, t, normal = self.ray_ellipsoid_intersection(
                        camera_pos, 
                        ray_dir, 
                        self.ellipsoids[e_idx].center, 
                        self.ellipsoids[e_idx].radii, 
                        self.ellipsoids[e_idx].rotation
                    )
                    
                    if hit and t < closest_t:
                        closest_hit = True
                        closest_t = t
                        closest_normal = normal
                        closest_color = self.ellipsoids[e_idx].color
                        closest_idx = e_idx
            
            # if there is an intersection, calculate the shading
            if closest_hit:
                # calculate the intersection point
                hit_point = camera_pos + closest_t * ray_dir
                
                # calculate the light direction and view direction
                light_dir = (self.light_position[None] - hit_point).normalized()
                view_dir = (camera_pos - hit_point).normalized()
                
                # calculate the half direction for specular reflection
                half_dir = (light_dir + view_dir).normalized()
                
                # calculate the attenuation of the light
                light_distance = ti.sqrt((self.light_position[None] - hit_point).dot(self.light_position[None] - hit_point))
                attenuation = 1.0 / (1.0 + 0.09 * light_distance + 0.032 * light_distance * light_distance)
                
                ambient_component = self.ambient * closest_color
                diff = max(0.0, closest_normal.dot(light_dir))
                diffuse_component = self.diffuse_strength * diff * closest_color * attenuation
                spec = ti.pow(max(0.0, closest_normal.dot(half_dir)), self.shininess)
                specular_component = self.specular_strength * spec * ti.Vector([1.0, 1.0, 1.0]) * attenuation
                
                shadow_factor = 1.0
                shadow_ray_origin = hit_point + closest_normal * 0.001  # avoid self-intersection
                shadow_ray_dir = light_dir

                # calculate the shadow factor
                for e_idx in range(self.num_ellipsoids):
                    if self.visible[e_idx] == 1 and e_idx != closest_idx:
                        shadow_hit, shadow_t, _ = self.ray_ellipsoid_intersection(
                            shadow_ray_origin,
                            shadow_ray_dir,
                            self.ellipsoids[e_idx].center,
                            self.ellipsoids[e_idx].radii,
                            self.ellipsoids[e_idx].rotation
                        )
                        
                        if shadow_hit:
                            shadow_distance = shadow_t * shadow_ray_dir.dot(shadow_ray_dir)
                            shadow_softness = ti.min(1.0, shadow_distance * 0.1)
                            shadow_factor = ti.min(shadow_factor, shadow_softness)
                
                color = ambient_component + \
                        (diffuse_component + specular_component) * shadow_factor
            
            color = ti.min(ti.max(color, 0.0), 1.0)

            self.pixels[i, j] = color  # update pixels

