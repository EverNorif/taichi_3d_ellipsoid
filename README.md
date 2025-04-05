# taichi_3d_ellipsoid

This repository provides a simple way to render ellipsoids using [taichi](https://www.taichi-lang.org/).

> For a very large number of ellipsoids, the current FPS is still relatively low. :(

## Installation

You can install this repository as a python package.

```bash
git clone https://github.com/EverNorif/taichi_3d_ellipsoid.git
cd taichi_3d_ellipsoid
pip install -e .
```

## Examples
In `run_example.py`, we provide two examples to demonstrate how to use the `EllipsoidRenderer`. You can run different examples by modifying the param `example_case` in it.

In Example 1, we randomly initialize various parameters of ellipsoids in a box and use GUI for visualization, where you can control the camera through keyboard and mouse. You can see the following effect like:

In Example 2, we load various parameters of ellipsoids from a 3DGS ply file, then specify the corresponding camera parameters to save the image. We provide `example.ply` [here](https://drive.google.com/file/d/17pQjk7sCkirzP6TBiMKyMVmGVxs5AHGe/view?usp=sharing).

Note that if you are using Mac platform, you should change the `ti.arch` to `ti.vulkan`, otherwise the saved image may be incomplete.

> Usually a 3DGS ply contains a very large number of particles, and currently the rendering FPS and speed are still very low.

