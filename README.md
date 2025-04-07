# taichi_3d_ellipsoid

This repository provides a simple way to render ellipsoids using [taichi](https://www.taichi-lang.org/).

## Installation

You can install this repository as a python package.

```bash
git clone https://github.com/EverNorif/taichi_3d_ellipsoid.git
cd taichi_3d_ellipsoid
pip install -e .
```

## Examples
In `run_example.py`, we provide four examples to demonstrate how to use the `EllipsoidRenderer`. You can run different examples by modifying the param `example_case` in it.

> In example 1 & 2, we use ray_tracing implement; and in example 3 & 4, we use rasterization implement.
>
> When setting `num_ellipsoids` to 50000, the rasterization implement can achieve 10x FPS faster than ray_tracing implement.

In Example 1 & 3, we randomly initialize various parameters of ellipsoids in a box and use GUI for visualization, where you can control the camera through keyboard and mouse. You can see the following effect like:

<p align="center">
  <img height="400" alt="example1" src="https://github.com/user-attachments/assets/7e94f7ee-96fc-4bce-ad47-4cfb59511074" />
</p>

In Example 2 & 4, we load various parameters of ellipsoids from a 3DGS ply file, then specify the corresponding camera parameters to save the image. We provide `example.ply` [here](https://drive.google.com/file/d/17pQjk7sCkirzP6TBiMKyMVmGVxs5AHGe/view?usp=sharing).

<p align="center">
  <img height="400" alt="example2" src="https://github.com/user-attachments/assets/21478c1f-95bc-412b-b001-b5924604a7bc" />
</p>

Note that if you are using Mac platform, you should change the `ti.arch` to `ti.vulkan`, otherwise the saved image may be incomplete.

> Usually a 3DGS ply contains a very large number of particles, and currently the rendering FPS and speed are still very low, especially when using ray_tracing implement.


## Visualization with taichi.js

We use [taichi.js](https://github.com/AmesingFlank/taichi.js) to visualize the ellipsoid rendering results [here](https://evernorif.github.io/taichi_3d_ellipsoid/). Related code can be found in the `docs` folder.

The basic demonstration can be found on [taichi.js_playground](https://taichi-js.com/playground/game-of-life). You can find the related source code in [StackBlitz](https://stackblitz.com/edit/taichi-js-game-of-life?file=index.js), which includes just one `index.html` and one `index.js`.
