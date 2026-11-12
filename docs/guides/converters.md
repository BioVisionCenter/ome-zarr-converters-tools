---
description: Ready-made OME-Zarr converters built on this library, by microscope and file format.
---

# Converters built on this library

**Ready-made converters, by format and microscope.**

Each project below is a set of Fractal tasks that reads one family of microscope
formats and writes OME-Zarr through this library. If your instrument is listed, start
there rather than here.

## fractal-uzh-converters

GitHub: [fractal-analytics-platform/fractal-uzh-converters](https://github.com/fractal-analytics-platform/fractal-uzh-converters)

Fractal tasks for high-content screening microscopes, plus generic TIFF folders.

| Microscope | Manufacturer |
|---|---|
| Operetta / Opera Phenix | Revvity |
| ScanR | Evident |
| CQ3K | Yokogawa |
| CellVoyager | Yokogawa |
| ImageXpress HCS.ai | Molecular Devices |
| TIFF folder, HCS plate | Any |
| TIFF folder, single images | Any |

## fractal-lif-converters

GitHub: [fractal-analytics-platform/fractal-lif-converters](https://github.com/fractal-analytics-platform/fractal-lif-converters)

Fractal tasks for Leica `.lif` files: HCS plates (single-position, multi-position and
mosaic wells) to OME-Zarr plates, and individual scenes (single or multi-position) to
standalone OME-Zarr images. Partial support for the autosave formats `xlef + .lof` and
`xlef + .tiff`.

## fractal-czi-converters

GitHub: [fractal-analytics-platform/fractal-czi-converters](https://github.com/fractal-analytics-platform/fractal-czi-converters)

Fractal tasks for Zeiss `.czi` files: HCS plates to OME-Zarr plates, and single,
multi-scene and mosaic scenes to standalone OME-Zarr images.

## fractal-nd2-converters

GitHub: [fractal-analytics-platform/fractal-nd2-converters](https://github.com/fractal-analytics-platform/fractal-nd2-converters)

Fractal tasks for Nikon `.nd2` files: HCS plates to OME-Zarr plates, and single and
multi-scene files to standalone OME-Zarr images.

## Adding a project

Built a converter on this library? Open an
[issue](https://github.com/BioVisionCenter/ome-zarr-converters-tools/issues) or a pull
request against this page.
