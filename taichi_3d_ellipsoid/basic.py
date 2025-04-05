import os
import taichi as ti
from typing import Tuple

@ti.dataclass
class Ellipsoid:
    center: ti.types.vector(3, ti.f32)
    radii: ti.types.vector(3, ti.f32)
    color: ti.types.vector(3, ti.f32)

@ti.data_oriented
class EllipsoidRenderer:
    def __init__(
        self,
        centers,
        radii,
        colors,
        arr_type: str="numpy",  # numpy or torch
        res:Tuple[int, int]=(1920, 1080),
        camera_pos:Tuple[float, float, float]=(0.0, 0.0, 5.0),
        camera_lookat:Tuple[float, float, float]=(0.0, 0.0, 0.0),
        camera_up:Tuple[float, float, float]=(0.0, 1.0, 0.0),
        fov:float=100.0,
        background_color:Tuple[float, float, float]=(0.05, 0.05, 0.05),
        ambient:float=0.2,
        diffuse_strength:float=0.6,
        headless:bool=False,
        ):
        
        # taichi init
        ti.init(arch=ti.gpu)

        # ellipsoid params
        self.ellipsoids = self._initialize_ellipsoids(centers, radii, colors, arr_type)
        self.num_ellipsoids = self.ellipsoids.shape[0]
        
        # camera params
        self.res_x, self.res_y = res # width, height
        self.start_camera_pos = camera_pos
        self.start_camera_lookat = camera_lookat
        self.start_camera_up = camera_up
        self.start_fov = fov
        
        # canvas pixels
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.res_x, self.res_y))
        self.background_color = ti.Vector(background_color)

        # material params
        self.ambient = ambient
        self.diffuse_strength = diffuse_strength

        # ggui window
        self.headless = headless
        self.window = ti.ui.Window(
            name="Taichi Ellipsoid Renderer", 
            res=(self.res_x, self.res_y), 
            show_window=not self.headless
        )
        self.canvas = self.window.get_canvas()
        
        self.camera = ti.ui.Camera()
        self._reset_camera_params()
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)

        # track camera params update
        self.cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_up = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_position = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_position[None] = ti.Vector(self.start_camera_pos)

    def _initialize_ellipsoids(self, centers, radii, colors, arr_type):
        assert len(centers) == len(radii) == len(colors), "centers, radii, colors must have the same length"
        ellipsoids = Ellipsoid.field(shape=len(centers))
        if arr_type == "numpy":
            ellipsoids.center.from_numpy(centers)
            ellipsoids.radii.from_numpy(radii)
            ellipsoids.color.from_numpy(colors)
        elif arr_type == "torch":
            ellipsoids.center.from_torch(centers)
            ellipsoids.radii.from_torch(radii)
            ellipsoids.color.from_torch(colors)
        else:
            raise ValueError(f"Unsupported array type: {arr_type}")
        
        return ellipsoids
    
    def _reset_camera_params(self):
        """reset self.camera."""
        self.camera.position(*self.start_camera_pos)
        self.camera.lookat(*self.start_camera_lookat)
        self.camera.up(*self.start_camera_up)
        self.camera.fov(self.start_fov)

    def _update_camera_params(self):
        """update camera params, will set the camera params to the fields, and update the light position equal to the camera position"""
        self.cam_pos[None] = self.camera.curr_position
        self.cam_lookat[None] = self.camera.curr_lookat
        self.cam_up[None] = self.camera.curr_up
        self.light_position[None] = self.camera.curr_position

    @ti.func
    def normalize(self, v):
        return v / ti.sqrt(v.dot(v))

    @ti.func
    def ray_ellipsoid_intersection(self, ray_origin, ray_dir, center, radii):
        """calculate the intersection between ray and ellipsoid"""
        # transform ray to ellipsoid local space
        oc = ray_origin - center
        
        # scale ray direction and origin, transform ellipsoid to unit sphere
        scaled_dir = ti.Vector([ray_dir[0] / radii[0], ray_dir[1] / radii[1], ray_dir[2] / radii[2]])
        scaled_oc = ti.Vector([oc[0] / radii[0], oc[1] / radii[1], oc[2] / radii[2]])
        
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
                
                # calculate the intersection point
                intersection_point = ray_origin + t * ray_dir
                
                # calculate the normal
                normal = ti.Vector([
                    (intersection_point[0] - center[0]) / (radii[0] * radii[0]),
                    (intersection_point[1] - center[1]) / (radii[1] * radii[1]),
                    (intersection_point[2] - center[2]) / (radii[2] * radii[2])
                ])
                normal = self.normalize(normal)
        
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
                hit, t, normal = self.ray_ellipsoid_intersection(
                    camera_pos, ray_dir, self.ellipsoids[e_idx].center, self.ellipsoids[e_idx].radii
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
                
                # light calculation
                light_dir = self.normalize(self.light_position[None] - hit_point)
                view_dir = self.normalize(camera_pos - hit_point)
                
                # calculate the light attenuation
                light_distance = ti.sqrt((self.light_position[None] - hit_point).dot(self.light_position[None] - hit_point))
                attenuation = 1.0 / (1.0 + 0.01 * light_distance + 0.001 * light_distance * light_distance)
                
                diff = max(0.0, closest_normal.dot(light_dir))
                ambient_component = self.ambient * closest_color
                diffuse_component = self.diffuse_strength * diff * closest_color * attenuation
                specular_component = 0.0
                
                color = ambient_component + diffuse_component + specular_component
                
                # don't process shadow each ellipsoid
                shadow_factor = 1.0
                shadow_origin = hit_point + closest_normal * 0.001  # avoid self-shadow
                color = ambient_component + (diffuse_component + specular_component) * shadow_factor
            
            color = ti.min(ti.max(color, 0.0), 1.0)

            self.pixels[i, j] = color  # update pixels
    
    def run_gui(self):
        assert not self.headless, "GUI mode is not supported in headless mode"
        print("""
        ========= start gui =========
        [w]: move camera forward
        [s]: move camera backward
        [a]: move camera left
        [d]: move camera right
        [q]: move camera down
        [e]: move camera up
        [r]: reset camera
        [mouse move]: rotate camera
        [esc]: exit
        ============================""")
        while self.window.running:
            if self.window.get_event(ti.ui.PRESS):
                if self.window.event.key == ti.ui.ESCAPE:
                    break
                elif self.window.event.key == 'r':
                    self._reset_camera_params()
            
            self.camera.track_user_inputs(self.window, movement_speed=0.05, hold_key=ti.ui.LMB)
            self._update_camera_params()

            self.render()
            self.canvas.set_image(self.pixels)
            self.window.show()
    
    def render_image(
        self,
        output_path: str,
        camera_pos: Tuple[float, float, float],
        camera_lookat: Tuple[float, float, float],
        camera_up: Tuple[float, float, float],
    ):
        self.camera.position(*camera_pos)
        self.camera.lookat(*camera_lookat)
        self.camera.up(*camera_up)

        self._update_camera_params()
        self.render()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ti.tools.imwrite(self.pixels, output_path)
