import os
import taichi as ti
from typing import Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

CONSOLE = Console(width=120)

@ti.dataclass
class Ellipsoid:
    center: ti.types.vector(3, ti.f32)
    radii: ti.types.vector(3, ti.f32)
    color: ti.types.vector(3, ti.f32)
    rotation: ti.types.matrix(3, 3, ti.f32)
    opacity: ti.f32

@ti.data_oriented
class EllipsoidRenderer:
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
        """
        Args:
            centers: list of centers of ellipsoids
            radii: list of radii of ellipsoids
            colors: list of colors of ellipsoids
            rotations: list of rotations of ellipsoids
            opacities: list of opacities of ellipsoids
            arr_type: numpy or torch
            res: resolution of the image
            camera_pos: position of the camera
            camera_lookat: lookat of the camera
            camera_up: up of the camera
            fov: field of view of the camera
            background_color: background color of the image 
            ambient: ambient light of the scene
            diffuse_strength: diffuse strength of the scene
            shininess: shininess of the scene
            specular_strength: specular strength of the scene
            headless: whether to run in headless mode
            opacity_limit: opacity limit of the ellipsoids
            ti_arch: taichi arch
            device: device to run on. Only used when arr_type is torch.
        """
        
        # taichi init
        ti.init(arch=ti_arch)
        self.arr_type = arr_type
        assert self.arr_type in ["numpy", "torch"], f"Unsupported array type: {self.arr_type}"
        self.device = device

        # ellipsoid params
        self.ellipsoids = self._initialize_ellipsoids(centers, radii, colors, rotations, opacities, self.arr_type)
        self.num_ellipsoids = self.ellipsoids.shape[0]
        self.visible = ti.field(ti.i32, self.num_ellipsoids)
        
        # camera params
        self.res_x, self.res_y = res # width, height
        self.start_camera_pos = camera_pos
        self.start_camera_lookat = camera_lookat
        self.start_camera_up = camera_up
        self.start_fov = fov
        self.opacity_limit = opacity_limit
        
        # canvas pixels
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.res_x, self.res_y))
        self.background_color = ti.Vector(background_color)

        # material params
        self.ambient = ambient
        self.diffuse_strength = diffuse_strength
        self.shininess = shininess
        self.specular_strength = specular_strength

        # ggui window
        self.headless = headless
        self.window = ti.ui.Window(
            name="Taichi Ellipsoid Renderer", 
            res=(self.res_x, self.res_y), 
            show_window=not self.headless,
            vsync=True
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

    def _initialize_ellipsoids(self, centers, radii, colors, rotations, opacities, arr_type):
        assert len(centers) == len(radii) == len(colors) == len(rotations) == len(opacities), \
            "centers, radii, colors, rotations, opacities must have the same length"
        ellipsoids = Ellipsoid.field(shape=len(centers))
        if arr_type == "numpy":
            ellipsoids.center.from_numpy(centers)
            ellipsoids.radii.from_numpy(radii)
            ellipsoids.color.from_numpy(colors)
            ellipsoids.rotation.from_numpy(rotations)
            ellipsoids.opacity.from_numpy(opacities)
        elif arr_type == "torch":
            ellipsoids.center.from_torch(centers)
            ellipsoids.radii.from_torch(radii)
            ellipsoids.color.from_torch(colors)
            ellipsoids.rotation.from_torch(rotations)
            ellipsoids.opacity.from_torch(opacities)
        
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
        self.light_position[None] = self.camera.curr_position  # NOTE: light position is the same as the camera position

    @ti.func
    def is_ellipsoid_visible(self, camera_pos, forward, right, up, half_width, half_height, center, radii):
        """Check if an ellipsoid is potentially visible in the camera's view frustum"""
        is_visible = True
        cam_to_center = center - camera_pos
        max_radius = ti.max(ti.max(radii[0], radii[1]), radii[2])
        distance_squared = cam_to_center.dot(cam_to_center)
        distance = ti.sqrt(distance_squared)
        
        # if the ellipsoid center is behind the camera and the distance is greater than the maximum radius, then it is not visible
        dot_product = cam_to_center.dot(forward)
        if dot_product < -max_radius:
            is_visible = False
        else:
            # check if the ellipsoid is outside the frustum
            up_component = cam_to_center.dot(up)
            right_component = cam_to_center.dot(right)
            
            # calculate the frustum boundaries (use dot_product as depth)
            frustum_height = ti.abs(dot_product) * half_height
            frustum_width = ti.abs(dot_product) * half_width
            
            # if the ellipsoid center plus the radius is outside the frustum, then it is not visible
            if ti.abs(up_component) > frustum_height + max_radius or \
               ti.abs(right_component) > frustum_width + max_radius:
                is_visible = False
        
        return is_visible

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
                normal = (rotation @ local_normal).normalized()
        
        return is_hit, t, normal
    
    @ti.kernel
    def render(self):
        """render the ellipsoids"""
        raise NotImplementedError("Subclasses must implement this method")

    def _print_controls(self):
        table = Table(show_header=False, box=box.ROUNDED, expand=False, border_style="cyan")
        table.add_column("Key", style="green bold")
        table.add_column("Action", style="yellow")
        
        table.add_row("[W]", "Move camera forward")
        table.add_row("[S]", "Move camera backward")
        table.add_row("[A]", "Move camera left")
        table.add_row("[D]", "Move camera right")
        table.add_row("[Q]", "Move camera down")
        table.add_row("[E]", "Move camera up")
        table.add_row("[R]", "Reset camera")
        table.add_row("Mouse move", "Rotate camera")
        table.add_row("[ESC]", "Exit")
        
        panel = Panel(
            table,
            title="[bold blue]Ellipsoid Renderer Controls[/bold blue]",
            subtitle="[italic]Press ESC to exit[/italic]",
            border_style="cyan",
            padding=(1, 2)
        )
        CONSOLE.print(panel)

    def run_gui(self):
        assert not self.headless, "GUI mode is not supported in headless mode"
        self._print_controls()
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

        if output_path is not None:
            directory = os.path.dirname(output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            ti.tools.imwrite(self.pixels, output_path)
        if self.arr_type == "torch":
            return self.pixels.to_torch(device=self.device)
        elif self.arr_type == "numpy":
            return self.pixels.to_numpy()
