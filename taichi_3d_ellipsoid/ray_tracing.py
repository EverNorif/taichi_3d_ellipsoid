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

    @ti.func
    def normalize(self, v):
        return v / ti.sqrt(v.dot(v))

    @ti.func
    def ray_ellipsoid_intersection(self, ray_origin, ray_dir, center, radii, rotation):
        """calculate the intersection between ray and ellipsoid"""
        # transform ray to ellipsoid local space
        oc = ray_origin - center
        
        # apply the inverse rotation
        # rotation is a forward rotation matrix, its transpose is the inverse rotation (because rotation matrix is orthogonal)
        inv_rotation = rotation.transpose()
        
        # transform ray direction and origin to ellipsoid local space
        local_dir = inv_rotation @ ray_dir
        local_oc = inv_rotation @ oc
        
        # scale ray direction and origin, transform ellipsoid to unit sphere
        scaled_dir = ti.Vector([
            local_dir[0] / radii[0],
            local_dir[1] / radii[1],
            local_dir[2] / radii[2]
        ])
        scaled_oc = ti.Vector([
            local_oc[0] / radii[0],
            local_oc[1] / radii[1],
            local_oc[2] / radii[2]
        ])
        
        # solve quadratic equation
        a = scaled_dir.dot(scaled_dir)
        b = 2.0 * scaled_oc.dot(scaled_dir)
        c = scaled_oc.dot(scaled_oc) - 1.0  # 1.0 is the square of the radius of the unit sphere
        
        discriminant = b * b - 4 * a * c
        
        # initialize return values
        is_hit = False
        t = 0.0
        normal = ti.Vector([0.0, 0.0, 0.0])
        
        # if discriminant is greater than or equal to 0, there is an intersection
        if discriminant >= 0:
            # calculate the nearest intersection
            t_temp = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            
            # if t is less than 0, the intersection point is in the opposite direction of the ray
            if t_temp < 0.0001:
                t_temp = (-b + ti.sqrt(discriminant)) / (2.0 * a)
            
            # if t is valid
            if t_temp >= 0.0001:
                is_hit = True
                t = t_temp
                
                # calculate the intersection point in local space
                local_intersection = local_oc + t * local_dir
                
                # calculate the normal in local space
                local_normal = ti.Vector([
                    local_intersection[0] / (radii[0] * radii[0]),
                    local_intersection[1] / (radii[1] * radii[1]),
                    local_intersection[2] / (radii[2] * radii[2])
                ])
                
                # transform normal to world space and normalize
                # note: normal transformation needs to use rotation matrix instead of its inverse
                normal = self.normalize(rotation @ local_normal)
        
        return is_hit, t, normal

    @ti.kernel
    def render(self):
        camera_pos = self.cam_pos[None]
        camera_lookat = self.cam_lookat[None]
        camera_up = self.cam_up[None]
        fov_radians = self.start_fov * ti.math.pi / 180.0
        
        # calculate the camera coordinate system
        forward = self.normalize(camera_lookat - camera_pos)
        right = self.normalize(ti.math.cross(forward, camera_up))
        up = self.normalize(ti.math.cross(right, forward))
        
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
            ray_dir = self.normalize(
                forward + 
                u * half_width * right + 
                v * half_height * up
            )
            
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
                light_dir = self.normalize(self.light_position[None] - hit_point)
                view_dir = self.normalize(camera_pos - hit_point)
                
                # calculate the half direction for specular reflection
                half_dir = self.normalize(light_dir + view_dir)
                
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

