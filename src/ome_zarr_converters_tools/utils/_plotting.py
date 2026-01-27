"""Tools for plotting tiled images in 3D using Plotly."""

from typing import Any

import numpy as np

from ome_zarr_converters_tools.core._tile_region import TiledImage

try:
    import plotly.graph_objects as go

    plotting_available = True
except ImportError:
    plotting_available = False
    go: Any


def _cube_from_points(p0, p1):
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )

    return vertices


def _plot_cube_wireframe(fig, p0, p1, color="red", name="Cube"):
    vertices = _cube_from_points(p0, p1)
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    # fig = go.Figure()

    for i, e in enumerate(edges):
        fig.add_trace(
            go.Scatter3d(
                x=[vertices[e[0], 0], vertices[e[1], 0]],
                y=[vertices[e[0], 1], vertices[e[1], 1]],
                z=[vertices[e[0], 2], vertices[e[1], 2]],
                mode="lines",
                line={"color": color, "width": 5},
                name=name if i == 0 else None,  # legend only once
                showlegend=(i == 0),
            )
        )
    return fig


def plot_tiled_images(
    tiled_image: TiledImage,
):
    """Plot tiled images in 3D using Plotly."""
    if not plotting_available:
        raise ImportError(
            "Plotly is not installed. Please install plotly to use this function."
        )
    color_scale = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
    ]

    fig = go.Figure()
    i = 0
    color = color_scale[0]
    pixel_size = tiled_image.pixelsize
    z_spacing = tiled_image.z_spacing
    for tile in tiled_image.regions:
        color = color_scale[i % len(color_scale)]
        i += 1
        roi = tile.roi
        x_slice = roi.get(axis_name="x")
        y_slice = roi.get(axis_name="y")
        z_slice = roi.get(axis_name="z")
        assert x_slice is not None and y_slice is not None and z_slice is not None
        x0, y0, z0 = x_slice.start, y_slice.start, z_slice.start
        x1 = x_slice.end
        y1 = y_slice.end
        z1 = z_slice.end
        assert x0 is not None and y0 is not None and z0 is not None
        assert x1 is not None and y1 is not None and z1 is not None
        _plot_cube_wireframe(
            fig,
            (x0, y0, z0),
            (x1, y1, z1),
            color=color,
            name=f"{tiled_image.name} - {tile.roi.name}",
        )

    fig.update_layout(
        scene={
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": pixel_size / z_spacing},
        },
        legend={"title": "Objects"},
    )
    fig.show()
