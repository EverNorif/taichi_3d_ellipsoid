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

<p align="center">
  <img height="300" alt="example1" src="https://github.com/user-attachments/assets/0226cd10-4ffb-44d9-89ae-f499c866b400" />
</p>


In Example 2, we load various parameters of ellipsoids from a 3DGS ply file, then specify the corresponding camera parameters to save the image. We provide `example.ply` [here](https://drive.google.com/file/d/17pQjk7sCkirzP6TBiMKyMVmGVxs5AHGe/view?usp=sharing).

<p align="center">
  <img height="300" alt="example2" src="https://github.com/user-attachments/assets/4812e97a-fe38-4ea9-bf75-965b547a9a6a" />
</p>


Note that if you are using Mac platform, you should change the `ti.arch` to `ti.vulkan`, otherwise the saved image may be incomplete.

> Usually a 3DGS ply contains a very large number of particles, and currently the rendering FPS and speed are still very low.


## Visualization with taichi.js

We use [taichi.js](https://github.com/AmesingFlank/taichi.js) to visualize the ellipsoid rendering results [here](https://evernorif.github.io/taichi_3d_ellipsoid/). Related code can be found in the `docs` folder.

The basic demonstration can be found on [taichi.js_playground](https://taichi-js.com/playground/game-of-life). You can find the related source code in [StackBlitz](https://stackblitz.com/edit/taichi-js-game-of-life?file=index.js), which includes just one `index.html` and one `index.js`.