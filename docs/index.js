let main = async () => {
  await ti.init();

  // get canvas element
  let htmlCanvas = document.getElementById('result_canvas');
  htmlCanvas.width = 1280;
  htmlCanvas.height = 720;
  let canvas = new ti.Canvas(htmlCanvas);
  
  // ellipsoid number
  const n_ellipsoids = 50;
  
  // ellipsoid data structure
  let ellipsoidType = ti.types.struct({
    center: ti.types.vector(ti.f32, 3),
    radii: ti.types.vector(ti.f32, 3),
    color: ti.types.vector(ti.f32, 3),
    rotation: ti.types.matrix(ti.f32, 3, 3),
    opacity: ti.f32
  });
  
  // create ellipsoid array
  let ellipsoids = ti.field(ellipsoidType, n_ellipsoids);
  let visible = ti.field(ti.i32, n_ellipsoids);
  
  // rendering parameters
  let res_x = htmlCanvas.width;
  let res_y = htmlCanvas.height;
  let camera_pos = ti.Vector.field(3, ti.f32, [1]);
  let camera_lookat = ti.Vector.field(3, ti.f32, [1]);
  let camera_up = ti.Vector.field(3, ti.f32, [1]);
  let light_position = ti.Vector.field(3, ti.f32, [1]);
  let fov = 60.0 * Math.PI / 180.0;
  let background_color = [0.05, 0.05, 0.05];
  let ambient = 0.2;
  let diffuse_strength = 0.6;
  let opacity_limit = 0.2;
  let shininess = 32.0;
  let specular_strength = 0.5;
  let shadow_bias = 0.001;
  let shadow_softness_factor = 0.1;
  
  // rendering result
  let pixels = ti.Vector.field(3, ti.f32, [res_x, res_y]);
  
  ti.addToKernelScope({
    n_ellipsoids,
    ellipsoids,
    visible,
    res_x,
    res_y,
    pixels,
    camera_pos,
    camera_lookat,
    camera_up,
    light_position,
    fov,
    background_color,
    ambient,
    diffuse_strength,
    opacity_limit,
    shininess,
    specular_strength,
    shadow_bias,
    shadow_softness_factor
  });

  // initialize ellipsoid data
  let initEllipsoids = ti.kernel(() => {
    for (let i of ti.range(n_ellipsoids)) {
      // random generate ellipsoid center position
      ellipsoids[i].center = [
        ti.random() * 5 - 2.5,
        ti.random() * 5 - 2.5,
        ti.random() * 5 - 2.5
      ];
      // random generate ellipsoid radii 
      let base_radius = ti.random();
      ellipsoids[i].radii = [
        base_radius,
        base_radius * (0.1 + ti.random() * 0.5),
        base_radius * (0.1 + ti.random() * 0.5)
      ];
      // random generate color
      ellipsoids[i].color = [
        ti.random(),
        ti.random(),
        ti.random()
      ];
      // random generate rotation matrix
      let angle_x = ti.random() * Math.PI * 2;
      let angle_y = ti.random() * Math.PI * 2;
      let angle_z = ti.random() * Math.PI * 2;
      // rotation matrix around X axis
      let rot_x = [
        [1.0, 0.0, 0.0],
        [0.0, Math.cos(angle_x), -Math.sin(angle_x)],
        [0.0, Math.sin(angle_x), Math.cos(angle_x)]
      ];
      // rotation matrix around Y axis
      let rot_y = [
        [Math.cos(angle_y), 0.0, Math.sin(angle_y)],
        [0.0, 1.0, 0.0],
        [-Math.sin(angle_y), 0.0, Math.cos(angle_y)]
      ];
      // rotation matrix around Z axis
      let rot_z = [
        [Math.cos(angle_z), -Math.sin(angle_z), 0.0],
        [Math.sin(angle_z), Math.cos(angle_z), 0.0],
        [0.0, 0.0, 1.0]
      ];
      
      // generate final rotation matrix
      ellipsoids[i].rotation = rot_z.matmul(rot_y).matmul(rot_x);
      // set opacity
      ellipsoids[i].opacity = 0.2 + 0.8 * ti.random();
    }
  });
  
  // reset camera parameters
  let resetCamera = ti.kernel(() => {
    // set camera position
    camera_pos[0] = [0.0, 0.0, 5.0];
    camera_lookat[0] = [0.0, 0.0, 0.0];
    camera_up[0] = [0.0, 1.0, 0.0];
    // set light position to camera position
    light_position[0] = camera_pos[0];
  });
  
  // cross product
  let cross = (a, b) => {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  };
  
  // ellipsoid and ray intersection
  let rayEllipsoidIntersection = (ray_origin, ray_dir, center, radii, rotation) => {
    // transform ray to ellipsoid local space
    let oc = [
      ray_origin[0] - center[0],
      ray_origin[1] - center[1],
      ray_origin[2] - center[2]
    ];
    
    // apply inverse rotation
    let inv_rotation = rotation.transpose();
    
    // transform ray direction and origin to ellipsoid local space
    let local_dir = inv_rotation.matmul(ray_dir);
    let local_oc = inv_rotation.matmul(oc);
    
    // scale ray direction and origin, transform ellipsoid to unit sphere
    let scaled_dir = [
      local_dir[0] / radii[0],
      local_dir[1] / radii[1],
      local_dir[2] / radii[2]
    ];
    
    let scaled_oc = [
      local_oc[0] / radii[0],
      local_oc[1] / radii[1],
      local_oc[2] / radii[2]
    ];
    
    // solve quadratic equation
    let a = scaled_dir[0] * scaled_dir[0] + scaled_dir[1] * scaled_dir[1] + scaled_dir[2] * scaled_dir[2];
    let b = 2.0 * (scaled_oc[0] * scaled_dir[0] + scaled_oc[1] * scaled_dir[1] + scaled_oc[2] * scaled_dir[2]);
    let c = scaled_oc[0] * scaled_oc[0] + scaled_oc[1] * scaled_oc[1] + scaled_oc[2] * scaled_oc[2] - 1.0;
    
    let discriminant = b * b - 4 * a * c;
    
    // initialize return values
    let is_hit = false;
    let t = 0.0;
    let normal = [0.0, 0.0, 0.0];
    
    // if discriminant is greater than or equal to 0, then there is an intersection
    if (discriminant >= 0) {
      // calculate the nearest intersection
      let t_temp = (-b - Math.sqrt(discriminant)) / (2.0 * a);
      
      // if t is less than 0, then the intersection is behind the ray
      if (t_temp < 0.0001) {
        t_temp = (-b + Math.sqrt(discriminant)) / (2.0 * a);
      }
      
      // if t is valid
      if (t_temp >= 0.0001) {
        is_hit = true;
        t = t_temp;
        
        // calculate the intersection in local space
        let local_intersection = [
          local_oc[0] + t * local_dir[0],
          local_oc[1] + t * local_dir[1],
          local_oc[2] + t * local_dir[2]
        ];
        
        // calculate the normal in local space
        let local_normal = [
          local_intersection[0] / (radii[0] * radii[0]),
          local_intersection[1] / (radii[1] * radii[1]),
          local_intersection[2] / (radii[2] * radii[2])
        ];
        
        // transform the normal to world space and normalize it
        let world_normal = rotation.matmul(local_normal);
        let length = Math.sqrt(world_normal[0] * world_normal[0] + world_normal[1] * world_normal[1] + world_normal[2] * world_normal[2]);
        normal = [world_normal[0] / length, world_normal[1] / length, world_normal[2] / length];
      }
    }
    
    return { is_hit, t, normal };
  };
  ti.addToKernelScope({rayEllipsoidIntersection});

  // View Frustum Culling
  let isEllipsoidVisible = (camera_pos, forward, right, up, half_width, half_height, center, radii) => {
    let visible = true;
    // calculate the vector from camera to ellipsoid center
    let cam_to_center = [
      center[0] - camera_pos[0],
      center[1] - camera_pos[1],
      center[2] - camera_pos[2]
    ];
    
    // find the maximum radius as the bounding sphere radius
    let max_radius = Math.max(radii[0], Math.max(radii[1], radii[2]));
    
    // calculate the distance from ellipsoid center to camera
    let distance = Math.sqrt(
      cam_to_center[0] * cam_to_center[0] + 
      cam_to_center[1] * cam_to_center[1] + 
      cam_to_center[2] * cam_to_center[2]
    );
    
    // if the ellipsoid center is behind the camera and the distance is greater than the maximum radius, then it is not visible
    let dot_product = cam_to_center[0] * forward[0] + 
                     cam_to_center[1] * forward[1] + 
                     cam_to_center[2] * forward[2];
    if (dot_product < -max_radius) {
      visible = false;
    } else {
      // check if the ellipsoid is outside the frustum
      // calculate the position of the ellipsoid center in the camera space
      let center_in_cam_space = [
        dot_product,
        cam_to_center[0] * up[0] + cam_to_center[1] * up[1] + cam_to_center[2] * up[2],
        cam_to_center[0] * right[0] + cam_to_center[1] * right[1] + cam_to_center[2] * right[2]
      ];
      
      // calculate the frustum boundaries
      let frustum_height = distance * half_height;
      let frustum_width = distance * half_width;
      
      // if the ellipsoid center plus the radius is outside the frustum, then it is not visible
      if (Math.abs(center_in_cam_space[1]) > frustum_height + max_radius ||
          Math.abs(center_in_cam_space[2]) > frustum_width + max_radius) {
        visible = false;
      }
    }
    return visible;
  };
  ti.addToKernelScope({isEllipsoidVisible});

  // render function
  let render = ti.kernel(() => {
    let cam_pos = camera_pos[0];
    let cam_lookat = camera_lookat[0];
    let cam_up = camera_up[0];
    
    // calculate camera coordinate system
    let forward = [
      cam_lookat[0] - cam_pos[0],
      cam_lookat[1] - cam_pos[1],
      cam_lookat[2] - cam_pos[2]
    ];
    let forward_length = Math.sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
    forward = [forward[0] / forward_length, forward[1] / forward_length, forward[2] / forward_length];
    
    let right = cross(forward, cam_up);
    let right_length = Math.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    right = [right[0] / right_length, right[1] / right_length, right[2] / right_length];
    
    let up = cross(right, forward);
    let up_length = Math.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
    up = [up[0] / up_length, up[1] / up_length, up[2] / up_length];
    
    // field of view parameters
    let half_height = Math.tan(fov / 2.0);
    let half_width = half_height * res_x / res_y;
    
    // iterate over each pixel
    for (let i of ti.range(res_x)) {
      for (let j of ti.range(res_y)) {
        // calculate normalized pixel coordinates (-1 to 1)
        let u = (i + 0.5) / res_x * 2 - 1;
        let v = (j + 0.5) / res_y * 2 - 1;
        
        // calculate ray direction
        let ray_dir = [
          forward[0] + u * half_width * right[0] + v * half_height * up[0],
          forward[1] + u * half_width * right[1] + v * half_height * up[1],
          forward[2] + u * half_width * right[2] + v * half_height * up[2]
        ];
        let ray_dir_length = Math.sqrt(ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1] + ray_dir[2] * ray_dir[2]);
        ray_dir = [ray_dir[0] / ray_dir_length, ray_dir[1] / ray_dir_length, ray_dir[2] / ray_dir_length];
        
        // default use background color
        let color = background_color;
        
        // track the nearest intersection
        let closest_hit = false;
        let closest_t = 1e10;
        let closest_normal = [0.0, 0.0, 0.0];
        let closest_color = [0.0, 0.0, 0.0];
        let closest_idx = -1;

        // preprocess frustum culling
        for (let e_idx of ti.range(n_ellipsoids)) {
          if (isEllipsoidVisible(
            cam_pos, 
            forward,
            right,
            up, 
            half_width, 
            half_height, 
            ellipsoids[e_idx].center, 
            ellipsoids[e_idx].radii
          )) {
            visible[e_idx] = 1;
          } else {
            visible[e_idx] = 0;
          }
        }

        // detect the intersection between ray and all ellipsoids
        for (let e_idx of ti.range(n_ellipsoids)) {
          if (ellipsoids[e_idx].opacity >= opacity_limit && visible[e_idx] == 1) {
            let result = rayEllipsoidIntersection(
              cam_pos, 
              ray_dir, 
              ellipsoids[e_idx].center, 
              ellipsoids[e_idx].radii, 
              ellipsoids[e_idx].rotation
            );
            
            if (result.is_hit && result.t < closest_t) {
              closest_hit = true;
              closest_t = result.t;
              closest_normal = result.normal;
              closest_color = ellipsoids[e_idx].color;
              closest_idx = e_idx;
            }
          }
        }
        
        // if there is an intersection, calculate shading
        if (closest_hit) {
          // calculate the intersection point
          let hit_point = [
            cam_pos[0] + closest_t * ray_dir[0],
            cam_pos[1] + closest_t * ray_dir[1],
            cam_pos[2] + closest_t * ray_dir[2]
          ];
          
          // calculate lighting
          let light_dir = [
            light_position[0][0] - hit_point[0],
            light_position[0][1] - hit_point[1],
            light_position[0][2] - hit_point[2]
          ];
          let light_dir_length = Math.sqrt(light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] + light_dir[2] * light_dir[2]);
          light_dir = [light_dir[0] / light_dir_length, light_dir[1] / light_dir_length, light_dir[2] / light_dir_length];
          
          let view_dir = [
            cam_pos[0] - hit_point[0],
            cam_pos[1] - hit_point[1],
            cam_pos[2] - hit_point[2]
          ];
          let view_dir_length = Math.sqrt(view_dir[0] * view_dir[0] + view_dir[1] * view_dir[1] + view_dir[2] * view_dir[2]);
          view_dir = [view_dir[0] / view_dir_length, view_dir[1] / view_dir_length, view_dir[2] / view_dir_length];
          
          // calculate half direction for specular
          let half_dir = [
            light_dir[0] + view_dir[0],
            light_dir[1] + view_dir[1],
            light_dir[2] + view_dir[2]
          ];
          let half_dir_length = Math.sqrt(half_dir[0] * half_dir[0] + half_dir[1] * half_dir[1] + half_dir[2] * half_dir[2]);
          half_dir = [half_dir[0] / half_dir_length, half_dir[1] / half_dir_length, half_dir[2] / half_dir_length];
          
          // calculate light attenuation
          let light_distance = Math.sqrt(
            (light_position[0][0] - hit_point[0]) * (light_position[0][0] - hit_point[0]) +
            (light_position[0][1] - hit_point[1]) * (light_position[0][1] - hit_point[1]) +
            (light_position[0][2] - hit_point[2]) * (light_position[0][2] - hit_point[2])
          );
          let attenuation = 1.0 / (1.0 + 0.09 * light_distance + 0.032 * light_distance * light_distance);
          
          // calculate ambient component
          let ambient_component = [
            ambient * closest_color[0],
            ambient * closest_color[1],
            ambient * closest_color[2]
          ];
          
          // calculate diffuse component
          let diff = Math.max(0.0, closest_normal[0] * light_dir[0] + closest_normal[1] * light_dir[1] + closest_normal[2] * light_dir[2]);
          let diffuse_component = [
            diffuse_strength * diff * closest_color[0] * attenuation,
            diffuse_strength * diff * closest_color[1] * attenuation,
            diffuse_strength * diff * closest_color[2] * attenuation
          ];
          
          // calculate specular component
          let spec = Math.pow(
            Math.max(0.0, closest_normal[0] * half_dir[0] + 
                         closest_normal[1] * half_dir[1] + 
                         closest_normal[2] * half_dir[2]), 
            shininess
          );
          let specular_component = [
            specular_strength * spec * attenuation,
            specular_strength * spec * attenuation,
            specular_strength * spec * attenuation
          ];
          
          // calculate shadow factor
          let shadow_factor = 1.0;
          let shadow_ray_origin = [
            hit_point[0] + closest_normal[0] * shadow_bias,
            hit_point[1] + closest_normal[1] * shadow_bias,
            hit_point[2] + closest_normal[2] * shadow_bias
          ];
          
          // check for shadow
          for (let e_idx of ti.range(n_ellipsoids)) {
            if (visible[e_idx] == 1 && e_idx != closest_idx) {
              let shadow_result = rayEllipsoidIntersection(
                shadow_ray_origin,
                light_dir,
                ellipsoids[e_idx].center,
                ellipsoids[e_idx].radii,
                ellipsoids[e_idx].rotation
              );
              
              if (shadow_result.is_hit) {
                let shadow_distance = shadow_result.t;
                let shadow_softness = Math.min(1.0, shadow_distance * shadow_softness_factor);
                shadow_factor = Math.min(shadow_factor, shadow_softness);
              }
            }
          }
          
          // combine components
          color = [
            ambient_component[0] + (diffuse_component[0] + specular_component[0]) * shadow_factor,
            ambient_component[1] + (diffuse_component[1] + specular_component[1]) * shadow_factor,
            ambient_component[2] + (diffuse_component[2] + specular_component[2]) * shadow_factor
          ];
        }
        
        // ensure color is in the range of 0 to 1
        color = [
          Math.max(0.0, Math.min(1.0, color[0])),
          Math.max(0.0, Math.min(1.0, color[1])),
          Math.max(0.0, Math.min(1.0, color[2]))
        ];
        
        // update pixel
        pixels[i, j] = color;
      }
    }
  });

  // handle keyboard input, change camera position
  document.addEventListener('keydown', async function (event) {
    // create a function to update camera position and target point
    const updateCamera = async (dx, dy, dz) => {
      // get current camera position and target point
      const pos = await camera_pos.get([0]);
      const lookat = await camera_lookat.get([0]);
      
      // create updated position array and target point array
      const newPos = [pos[0] + dx, pos[1] + dy, pos[2] + dz];
      const newLookat = [lookat[0] + dx, lookat[1] + dy, lookat[2] + dz];
      
      // set new camera position and target point
      camera_pos.set([0], newPos);
      camera_lookat.set([0], newLookat);
      
      // update light position to be the same as camera position
      light_position.set([0], newPos);
    };

    if (event.key.toUpperCase() === 'W') {
      // move camera forward
      await updateCamera(0, 0, -0.1);
    }
    if (event.key.toUpperCase() === 'S') {
      // move camera backward
      await updateCamera(0, 0, 0.1);
    }
    if (event.key.toUpperCase() === 'A') {
      // move camera left
      await updateCamera(-0.1, 0, 0);
    }
    if (event.key.toUpperCase() === 'D') {
      // move camera right
      await updateCamera(0.1, 0, 0);
    }
    if (event.key.toUpperCase() === 'Q') {
      // move camera down
      await updateCamera(0, -0.1, 0);
    }
    if (event.key.toUpperCase() === 'E') {
      // move camera up
      await updateCamera(0, 0.1, 0);
    }
    if (event.key.toUpperCase() === 'R') {
      // reset camera
      resetCamera();
    }
  });

  // initialize ellipsoids and camera
  initEllipsoids();
  resetCamera();
  
  // render and display
  async function frame() {
    if (window.shouldStop) {
      return;
    }
    render();
    canvas.setImage(pixels);
    requestAnimationFrame(frame);
  }
  await frame();
};
// This is just because StackBlitz has some weird handling of external scripts.
// Normally, you would just use `<script src="https://unpkg.com/taichi.js/dist/taichi.umd.js"></script>` in the HTML
const script = document.createElement('script');
script.addEventListener('load', async function () {
  await main();
  var h2 = document.getElementById('hint');
  h2.innerHTML = 'Try W/A/S/D/Q/E to move the camera, and R to reset the camera position';
});
script.src = 'https://unpkg.com/taichi.js/dist/taichi.umd.js';
// Append to the `head` element
document.head.appendChild(script);
