"""
taichi_3d_ellipsoid: render 3d ellipsoid with taichi.

author: EverNorif
version: 0.1.0
"""

__version__ = "0.1.0" 

from .ray_tracing import EllipsoidRayTracingRenderer
from .rasterization import EllipsoidRasterizationRenderer
from .basic import EllipsoidRenderer