---
description: Projects that use ome-zarr-converters-tools to convert microscopy formats to OME-Zarr.
---

# Downstream converters

The following projects use `ome-zarr-converters-tools` to build OME-Zarr converters for different microscopy formats.

## [fractal-uzh-converters](https://github.com/fractal-analytics-platform/fractal-uzh-converters)

A collection of Fractal tasks for converting HCS plate data from various high-content screening microscopes to OME-Zarr.

| Microscope | Manufacturer |
|---|---|
| Operetta / Opera Phenix | Revvity |
| ScanR | Evident |
| CQ3K | Yokogawa |
| CellVoyager | Yokogawa |
| ImageXpress HCS.ai | Molecular Devices |
| TIFF (HCS plate) | Any |
| TIFF (single images) | Any |

## [fractal-lif-converters](https://github.com/fractal-analytics-platform/fractal-lif-converters)

Fractal tasks for converting Leica `.lif` files to OME-Zarr. Supports plate layouts (single-position, multi-position, and mosaic) and standalone scene conversions. Partial support for autosave in format `xlef + .lof` and `xlef + .tiff`.
This converter supports both converting Leica files containing HCS plate data to OME-Zarr HCS plates, as well as converting individual scenes (single or multi-position) to standalone OME-Zarr images.

## [fractal-czi-converters](https://github.com/fractal-analytics-platform/fractal-czi-converters)

Fractal tasks for converting Zeiss `.czi` files to OME-Zarr. This converter supports both converting Zeiss files containing HCS plate data to OME-Zarr HCS plates, as well as converting single scenes, multi-scenes and mosaic scenes to standalone OME-Zarr images.

## [fractal-nd2-converters](https://github.com/fractal-analytics-platform/fractal-nd2-converters)

Fractal tasks for converting Nikon `.nd2` files to OME-Zarr. This converter supports both converting Nikon files containing HCS plate data to OME-Zarr HCS plates, as well as converting single scenes and multi-scenes to standalone OME-Zarr images.
