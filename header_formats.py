"""A collection of methods for creating header-only files for different formats."""

import h5py
import numpy as np
from crc32c import crc32c
from libtiff import libtiff_ctypes as libtiffc
from libtiff import TIFF
import shutil
import os
import tensorstore as ts


def hdf5_header_only(
    filename: str,
    shape: tuple,
    dtype: np.dtype,
    *,
    chunks: tuple | None = None,
    header_size: int | None = None,
    userblock_size: int | None = None,
    dataset_name: str = "data",
    zarr_index: bool = False,
) -> None:
    """Creates an HDF5 file with only a header, no data.

    Args:
        filename (str): The name of the HDF5 file to create.
        shape (tuple): The shape of the dataset.
        dtype (str or np.dtype): The data type of the dataset.
        chunks (tuple, optional): The chunk shape for the dataset. Defaults to None.
        header_size (int, optional): The size of the HDF5 header. Defaults to None.
        userblock_size (int, optional): The size of the user block. Defaults to None.
        dataset_name (str, optional): The name of the dataset. Defaults to "data".
        zarr_index (bool, optional): Whether to create a zarr index. Defaults to False.
    """
    if not header_size:
        if chunks:
            nchunks = np.prod(-(shape // -np.array(chunks)))
            header_size = 1024 + nchunks * 16
            header_size = 2 ** np.ceil(np.log2(header_size)).astype("int")
        else:
            header_size = 2048

    dtype = np.dtype(dtype)

    with h5py.File(
        filename,
        "w",
        meta_block_size=header_size,
        userblock_size=userblock_size,
        libver="v112",
    ) as h5f:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        if chunks:
            dcpl.set_chunk(chunks)
        type_id = h5py.h5t.py_create(dtype, logical=1)
        space_id = h5py.h5s.create_simple(shape, shape)
        h5data = h5py.h5d.create(
            h5f.id, dataset_name.encode("utf-8"), type_id, space_id, dcpl
        )

        if zarr_index:
            offsets_and_nbytes = get_hdf5_chunk_offsets_and_bytes(h5data)
            # Older versions of HDF5 than 1.14.3 do not incorporate userblock_size into offsets
            # https://github.com/HDFGroup/hdf5/issues/3003
            if userblock_size and h5py.h5.get_libversion() < (1, 14, 3):
                offsets_and_nbytes[:, 0] += userblock_size
            zindex_bytes = offsets_and_nbytes.tobytes()
            zindex_bytes += int.to_bytes(crc32c(zindex_bytes), 4, "little")
            zindex_dataset = h5f.create_dataset("zarrindex", (nchunks * 16 + 4,), "uint8")
            zindex_dataset[:] = np.frombuffer(zindex_bytes, dtype="uint8")


def simple_zarr_index(h5parent: h5py.File | h5py.Group, h5dataset: h5py.Dataset) -> None:
    """Creates a simple zarr index for an HDF5 dataset.

    Args:
        h5parent (h5py.File or h5py.Group): The parent HDF5 object.
        h5dataset (h5py.Dataset): The HDF5 dataset to index.
    """
    if type(h5dataset) == h5py.Dataset:
        h5dataset = h5dataset.id

    dtype = h5dataset.dtype
    shape = h5dataset.shape
    chunks = h5dataset.get_create_plist().get_chunk()
    if chunks:
        chunk_offsets_and_nbytes = get_hdf5_chunk_offsets_and_bytes(h5dataset)
        nchunks = chunk_offsets_and_nbytes.shape[0]
    else:
        nchunks = 1
        chunk_offsets = np.array([header_size], dtype="uint64")
        chunk_nbytes = np.array([np.prod(shape) * dtype.itemsize], dtype="uint64")
        chunk_offsets_and_nbytes = np.stack((chunk_offsets, chunk_nbytes), axis=1)

    zindex_bytes = chunk_offsets_and_nbytes.tobytes()
    zindex_bytes += int.to_bytes(crc32c(zindex_bytes), 4, "little")
    zindex_dataset = h5parent.create_dataset("zarrindex", (nchunks * 16 + 4,), "uint8")
    zindex_dataset[:] = np.frombuffer(zindex_bytes, dtype="uint8")


def get_hdf5_chunk_offsets_and_bytes(h5dataset: h5py.Dataset) -> np.ndarray:
    """Gets the chunk offsets and byte lengths for an HDF5 dataset.

    Args:
        h5dataset (h5py.Dataset): The HDF5 dataset.

    Returns:
        np.ndarray: A 2D numpy array where each row contains the offset and length of a chunk.
    """
    if type(h5dataset) == h5py.Dataset:
        h5dataset = h5dataset.id

    shard_shape = h5dataset.shape
    chunk_shape = h5dataset.get_create_plist().get_chunk()
    nchunks = np.prod(-(shard_shape // -np.array(chunk_shape)))

    chunk_info = {}
    h5dataset.chunk_iter(lambda x: chunk_info.__setitem__(x.chunk_offset, x))
    offsets_and_lengths = np.empty((nchunks, 2), dtype="uint64")

    # number of chunks per axes
    nc = tuple(map(lambda s, c: s // c, shard_shape, chunk_shape))
    coordinates = [idx for idx in np.ndindex(nc)]

    for i, coords in enumerate(coordinates):
        chunk_offset = tuple(map(lambda x, c: x * c, coords, chunk_shape))
        ci = chunk_info.get(chunk_offset, None)
        if ci is None:
            # missing chunk
            offsets_and_lengths[i, 0] = 0xFFFFFFFFFFFFFFFF
            offsets_and_lengths[i, 1] = 0xFFFFFFFFFFFFFFFF
        else:
            offsets_and_lengths[i, 0] = ci.byte_offset
            offsets_and_lengths[i, 1] = ci.size
    return offsets_and_lengths


def write_tiff_header(
    filename: str, shape: tuple, dtype: np.dtype, *, chunks: tuple | None = None
) -> tuple[int, int]:
    """Writes a TIFF header to a file.

    Args:
        filename (str): The name of the TIFF file.
        shape (tuple): The shape of the image.
        dtype (str or np.dtype): The data type of the image.
        chunks (tuple, optional): The tile shape. Defaults to None.

    Returns:
        tuple: A tuple containing the size of the header and the offset to the first tile.
    """
    dtype = np.dtype(dtype)
    bits = dtype.itemsize * 8

    tif = TIFF.open(filename, "w")
    tif.SetField("IMAGELENGTH", shape[0])
    tif.SetField("IMAGEWIDTH", shape[1])
    tif.SetField("BITSPERSAMPLE", bits)
    tif.SetField("SAMPLEFORMAT", libtiffc.SAMPLEFORMAT_UINT)
    tif.SetField("PHOTOMETRIC", libtiffc.PHOTOMETRIC_MINISBLACK)
    tif.SetField("ORIENTATION", libtiffc.ORIENTATION_TOPLEFT)
    tif.SetField("PLANARCONFIG", libtiffc.PLANARCONFIG_CONTIG)
    if chunks:
        tif.SetField("TILELENGTH", chunks[0])
        tif.SetField("TILEWIDTH", chunks[1])
        if len(chunks) > 2:
            tif.SetField("TILEDEPTH", chunks[2])
        tif.WriteDirectory()
        tif.SetDirectory(0)
    else:
        tif.SetField("TILELENGTH", shape[0])
        tif.SetField("TILEWIDTH", shape[1])
        if len(shape) > 2:
            tif.SetField("TILEDEPTH", shape[2])
        tif.WriteDirectory()
        tif.SetDirectory(0)
    tif.close()

    with open(filename, "r") as f:
        f.seek(0, 2)
        sz = f.tell()

    with open(filename, "ab+") as f:
        f.seek(sz, 0)
        f.write(np.zeros((1024 + 2048 - sz,), dtype="uint8"))

    tif = TIFF.open(filename, "a+")
    tif.SetDirectory(0)
    A = np.full(shape, 0xABCD, dtype="uint16")
    tif.write_tiles(A)
    tif.SetDirectory(0)
    first_offset = tif.GetField("TileOffsets")
    tif.close()

    return sz, first_offset.value


def tiff_hdf5_zarr(
    filename: str, shape: tuple, dtype: np.dtype, *, chunks: tuple | None = None
) -> None:
    """Creates a file with a TIFF header, an HDF5/Zarr header, and no data.

    Args:
        filename (str): The name of the file to create.
        shape (tuple): The shape of the data.
        dtype (str or np.dtype): The data type of the data.
        chunks (tuple, optional): The chunk shape. Defaults to None.
    """
    userblock_size = 1024
    hdf5_header_only(
        filename, shape, dtype, zarr_index=True, chunks=chunks, userblock_size=userblock_size
    )
    write_tiff_header("temp.tiff", shape, dtype, chunks=(128, 128))

    with open("temp.tiff", "rb") as f:
        userblock = f.read(userblock_size)

    with open(filename, "rb+") as f:
        f.write(userblock)


def read_header_footer(
    filename: str = "test.tiff.hdf5.zarr", header_size: int = 3072, footer_size: int = 68
) -> tuple[bytes, bytes]:
    """Reads the header and footer of a file.

    Args:
        filename (str, optional): The name of the file. Defaults to "test.tiff.hdf5.zarr".
        header_size (int, optional): The size of the header. Defaults to 3072.
        footer_size (int, optional): The size of the footer. Defaults to 68.

    Returns:
        tuple: A tuple containing the header and footer as bytes.
    """
    with open(filename, "rb") as f:
        header = f.read(header_size)
        f.seek(-footer_size, 2)
        footer = f.read(footer_size)
    return header, footer


def create_zarr3(path: str) -> None:
    """Creates a Zarr v3 dataset using tensorstore.

    Args:
        path (str): The path to the Zarr dataset.
    """
    ds = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": path},
            "metadata": {
                "shape": [256, 256],
                "data_type": "uint16",
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {"chunk_shape": [128, 128]},
                    }
                ],
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()


def run_demo() -> None:
    """Runs a demonstration of the header creation and file manipulation functions."""
    tiff_hdf5_zarr("test.tiff.hdf5.zarr", (256, 256), "uint16", chunks=(128, 128))

    create_zarr3("demo/test.zarr")

    header, footer = read_header_footer("test.tiff.hdf5.zarr")
    A = np.full((128, 128), 0, dtype="uint16")
    B = np.full((128, 128), 2**14 - 2, dtype="uint16")
    C = np.full((128, 128), 2 * 2**14 - 2, dtype="uint16")
    D = np.full((128, 128), 3 * 2**14 - 2, dtype="uint16")

    os.makedirs("demo/test.zarr/c/0/")

    with open("demo/demo.hdf5.zarr.tiff", "wb") as f:
        f.write(header)
        f.write(A)
        f.write(B)
        f.write(C)
        f.write(D)
        f.write(footer)

    shutil.copyfile("demo/demo.hdf5.zarr.tiff", "demo/demo.tiff")
    shutil.copyfile("demo/demo.hdf5.zarr.tiff", "demo/demo.h5")
    shutil.copyfile("demo/demo.hdf5.zarr.tiff", "demo/test.zarr/c/0/0")
