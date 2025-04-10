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
    let near_plane = 0.1;
    let far_plane = 100.0;
    
    // rendering result
    let pixels = ti.Vector.field(3, ti.f32, [res_x, res_y]);

    // depth buffer
    let depth_buffer = ti.field(ti.f32, [res_x, res_y]);
    
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
      shadow_softness_factor,
      near_plane,
      far_plane,
      depth_buffer
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
      let viewport_height = 2.0 * half_height;
      let viewport_width = 2.0 * half_width;

      // clear the buffer
      for (let i of ti.range(res_x)) {
        for (let j of ti.range(res_y)) {
          pixels[i, j] = background_color;
          depth_buffer[i, j] = far_plane;
        }
      }

      // preprocess frustum culling
      for (let e_idx of ti.range(n_ellipsoids)) {
        if (ellipsoids[e_idx].opacity >= opacity_limit) {
          if (isEllipsoidVisible(
            cam_pos, forward, right, up, half_width, half_height,
            ellipsoids[e_idx].center, ellipsoids[e_idx].radii)) {
            visible[e_idx] = 1;
          } else {
            visible[e_idx] = 0;
          }
        } else {
          visible[e_idx] = 0;
        }
      }

      // rasterization
      for (let e_idx of ti.range(n_ellipsoids)) {
        if (ellipsoids[e_idx].opacity >= opacity_limit && visible[e_idx] == 1) {
          let center = ellipsoids[e_idx].center;
          let radii = ellipsoids[e_idx].radii;
          let rotation = ellipsoids[e_idx].rotation;

          // calculate the distance to the camera
          let to_center = center - cam_pos;
          let depth = to_center.dot(forward);

          if (depth > 0) {
            let color = [0.0, 0.0, 0.0];
            // calculate the screen space position
            let screen_x = to_center.dot(right) / depth * res_y / viewport_height + res_x / 2;
            let screen_y = to_center.dot(up) / depth * res_y / viewport_height + res_y / 2;

            // calculate the screen space radius, considering the maximum size of the ellipsoid
            let max_scale = ti.max(radii[0], ti.max(radii[1], radii[2]));
            let screen_radius = ti.sqrt(
              (max_scale * right).norm() ** 2 + 
              (max_scale * up).norm() ** 2
            ) / depth * res_y / viewport_height;

            // calculate the bounding box range
            let min_x = ti.max(0, ti.i32(screen_x - screen_radius - 2));
            let min_y = ti.max(0, ti.i32(screen_y - screen_radius - 2));
            let max_x = ti.min(res_x, ti.i32(screen_x + screen_radius + 2));
            let max_y = ti.min(res_y, ti.i32(screen_y + screen_radius + 2));

            for (let i_idx of ti.range(max_x-min_x)) {
              for (let j_idx of ti.range(max_y-min_y)) {
                // calculate the ray direction
                let i = i_idx + min_x;
                let j = j_idx + min_y;
                let u = (i - res_x / 2) * viewport_width / res_x;
                let v = (j - res_y / 2) * viewport_height / res_y;
                let ray_dir = (forward + u * right + v * up).normalized();
                
                let result = rayEllipsoidIntersection(
                  cam_pos, ray_dir, center, radii, rotation);

                if (result.is_hit) {
                  let view_space_depth = (cam_pos + ray_dir * result.t - cam_pos).dot(forward);
                  let pixel_depth = (view_space_depth - near_plane) / (far_plane - near_plane);

                  let old_depth = ti.atomicMin(depth_buffer[i, j], pixel_depth);
                  if (pixel_depth <= old_depth) {
                    let light_to_point = (cam_pos + ray_dir * result.t - light_position[0]).normalized();

                    let ambient_component = ambient * ellipsoids[e_idx].color;
                    let diffuse = max(result.normal.dot(-light_to_point), 0.0);
                    let diffuse_component = diffuse_strength * diffuse * ellipsoids[e_idx].color;

                    let half_vec = (-light_to_point + (-ray_dir)).normalized();
                    let specular = pow(max(result.normal.dot(half_vec), 0.0), shininess);
                    let specular_component = specular_strength * specular * [1.0, 1.0, 1.0];

                    let color = ambient_component + diffuse_component + specular_component;
                    // update pixel
                    pixels[i, j] = color;
                  }
                } 
              }
            }
          }
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

window.initRenderer = async () => {
  await main();
  return {
    cleanup: () => {
      // 清理资源
      if (window.ti) {
        window.ti.reset();
      }
    }
  };
};
  