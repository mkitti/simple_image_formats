# Simple image formats: Combining TIFF, HDF5, and Zarr

This repository contains code that demonstrates how TIFF, HDF5, and Zarr version 3 shards can be combined into a single file.

[View the notebook](https://github.com/mkitti/simple_image_formats/blob/main/header_formats.ipynb)

## Abstract

![XKCD Standards](https://imgs.xkcd.com/comics/standards.png)

"Situation: There are 3 competing standards" the last card of a popular [XKCD cartoon (#927)](https://xkcd.com/927) might read if applied to recent microscopy bioimaging formats.
TIFF, HDF5, and Zarr have all been used to store images as part of popular standards and formats (OME-TIFF, BDV-HDF5, OME-Zarr).
The cartoon humorously points out the tendency to create new standards while discounting prior efforts.
To combat the proliferation of formats I examine similarities between TIFF, HDF5, and Zarr shard containers.
I then exploit them to create a combined data container that is simultaneously a TIFF file, a HDF5 file, and a Zarr version 3 shard without duplicating the image pixel or volumetric voxel data.
This combined format is compatible with multiple viewers and image analysis pipelines. Additionally, the techniques involved provide a path to convert between the formats with minimal processing or overhead.
In practice, the combined format avoids redundant copies of data while providing great utility to the microscope user.
The combined format is a great candidate for a microscope acquisition format as it satisfies both short term needs
to view microscope output in traditional viewers while integrating into next generation image analysis pipelines.

## Outline

First, note the following properties of the three formats:
* The [TIFF specification ](https://download.osgeo.org/libtiff/doc/TIFF6.pdf) only requires a few bytes at the beginning of the file.
* The [HDF5 specification](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t2.html#subsec_fmt2_boot_super) allows for a user block to be present at the beginning of the file while allowing the HDF5 header to exist at 512 bytes, 1024 bytes, 2048 bytes, or some further doubling of bytes.
* The [Zarr version 3 sharding codec specification](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html) allows for the chunk index of shards to exist at the end of the file.

This suggests the following file structure:

| TIFF (1) |
| ---- |
| HDF5 (2) |
| Data (3) |
| Zarr (4) |

1. TIFF Header and Image File Directory points to tiles or cubes in the data (via [GeoTIFF](https://www.earthdata.nasa.gov/about/esdis/esco/standards-practices/geotiff) tag extensions).
2. HDF5 Header and Fixed Array Data Block points to the same tiles or cubes in the data
3. Image pixel or voxel data
4. Zarr v3 sharding codec index pointing to the same tile or cubes in the data.
