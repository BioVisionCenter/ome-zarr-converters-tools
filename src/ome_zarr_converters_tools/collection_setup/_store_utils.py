from zarr.storage import FsspecStore, LocalStore

ConverterStorageType = LocalStore | FsspecStore


def concat_storage(
    *,
    store: ConverterStorageType,
    path: str,
) -> ConverterStorageType:
    """Concatenate a path to a Zarr store.

    This is a helper function to create a new store without
    the zarr api creating json files at each level.

    Args:
        store: The base Zarr store.
        path: The path to concatenate.

    Returns:
        The concatenated Zarr store.
    """
    if isinstance(store, LocalStore):
        cat_path = store.root / path
        return LocalStore(cat_path)
    elif isinstance(store, FsspecStore):
        raise NotImplementedError(
            "Concatenation for FsspecStore is not implemented yet."
        )
    raise ValueError(f"Unsupported store type {type(store)}")
