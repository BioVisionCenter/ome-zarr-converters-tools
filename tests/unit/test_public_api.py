"""Guards for the public API surface (frozen at v1.0.0)."""

import importlib

import ome_zarr_converters_tools as pkg

# Primary entry points that must stay importable from the package root.
_ROOT_ENTRY_POINTS = [
    "hcs_images_from_dataframe",
    "single_images_from_dataframe",
    "tiled_image_from_tiles",
    "build_dummy_tile",
    "ImageLoaderInterface",
    "DefaultImageLoader",
    "ConverterOptions",
    "TilingStrategy",
    "AutoTiling",
    "SnapToGridTiling",
    "SnapToCornersTiling",
    "InplaceTiling",
    "NoTiling",
    "StagePositionCorrections",
    "WriterMode",
]


def test_root_all_names_resolve() -> None:
    for name in pkg.__all__:
        assert hasattr(pkg, name), f"{name} in __all__ but not importable"


def test_root_all_is_sorted_and_unique() -> None:
    assert pkg.__all__ == sorted(pkg.__all__)
    assert len(pkg.__all__) == len(set(pkg.__all__))


def test_primary_entry_points_at_root() -> None:
    for name in _ROOT_ENTRY_POINTS:
        assert name in pkg.__all__, f"{name} should be a documented root export"


def test_subpackage_all_names_resolve() -> None:
    for sub in ("core", "models", "pipelines", "fractal", "testing"):
        module = importlib.import_module(f"ome_zarr_converters_tools.{sub}")
        for name in module.__all__:
            assert hasattr(module, name), f"{sub}.{name} in __all__ but missing"
