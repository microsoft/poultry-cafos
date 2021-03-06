"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

This script generates imagery based features for each predicted _polygon_ generated by
`run_postprocessing.py`. For example, if `run_postprocessing.py` creates an output that
contains 7 million polygons from input imagery across the entire USA, then this script
will crop out the corresponding imagery for each of the 7 million polygons,
compute a feature vector from that imagery, and save the feature vectors to file. Note:
this will only consider pixels that are *within* the predicted polygons.

The two types of feature vectors that can be computed are:
- "spectral-histograms" - a count of the different spectral values across the 4 bands of
    NAIP imagery. There are 256 unique spectral values per bands, so this results in a
    vector of size 256*4.
- "cluster-histograms" - a count of the _quantized_ color values within the predicted
    polygon. Each pixel contains 4 channels, (R, G, B, NIR). We quantize each of these
    values into 4 bins by performing integer division by 64. This gives 4 unique values
    for R, G, B, and NIR. There are now 256 unique "colors" and we report a count of
    these.
"""
import argparse
import os
import time
from collections import defaultdict

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.mask
import shapely
import shapely.geometry
import shapely.ops

from cafo import utils

os.environ.update(utils.RASTERIO_BEST_PRACTICES)

BASE_4_ENCODING = np.array([4 ** 3, 4 ** 2, 4 ** 1, 4 ** 0])

parser = argparse.ArgumentParser(description="CAFO model training script")
parser.add_argument(
    "--input_fn",
    type=str,
    required=True,
    help="The path to a set of predictions in GPKG format",
)
parser.add_argument(
    "--output_fn",
    type=str,
    required=True,
    help="The output filename (should end with .npz)",
)
parser.add_argument(
    "--feature_type",
    choices=["spectral-histograms", "cluster-histograms"],
    required=True,
    help="The type of features to extract from each sample. 'spectral-histograms' give"
    + " a 256x4 sized vector of counts for each spectral value over all bands while"
    + " 'cluster-histogram' give a 256 sized vector of counts of quantized colors over"
    + " the 4 bands.",
)
args = parser.parse_args()


def filter_urls(urls, valid_imagery_set):
    for url in urls:
        if url.split("/")[7] in valid_imagery_set:
            return url
    raise ValueError("No match")


def main():

    assert os.path.exists(args.input_fn)
    assert not os.path.exists(args.output_fn)
    assert args.output_fn.endswith(".npz")

    # Load all of the input samples
    prediction_shapes = []
    prediction_geoms = []
    prediction_urls = []
    with fiona.open(args.input_fn) as f:
        for row in f:
            prediction_shapes.append(shapely.geometry.shape(row["geometry"]))
            prediction_geoms.append(row["geometry"])
            prediction_urls.append(row["properties"]["image_url"])
    for url in prediction_urls:
        assert url.startswith("https://naipblobs.blob.core.windows.net")

    # Create a mapping between samples and imagery tiles. I.e. which samples belong to
    # which imagery tile. If we did not pre-compute this mapping, then this script would
    # be incredibly inefficient as it would requiring loading each imagery tile many
    # times.
    # check to make sure that the URLs match up with what the NAIPTileIndex knows about
    tiles_to_geom_idxs = defaultdict(list)
    print("Computing sample to tile mapping")
    for i, url in enumerate(prediction_urls):
        if i % 10000 == 0:
            print(f"{i / len(prediction_geoms) * 100:0.4f}%")
        tiles_to_geom_idxs[url].append(i)

    # Compute features per sample
    if args.feature_type == "spectral-histograms":
        histograms = np.zeros((len(prediction_shapes), 4, 256), dtype=int)
    elif args.feature_type == "cluster-histograms":
        histograms = np.zeros((len(prediction_shapes), 256), dtype=int)

    bad_indices = []
    count = 0
    tic = time.time()
    for url, geom_idxs in tiles_to_geom_idxs.items():
        if len(geom_idxs) > 0:
            with rasterio.open(url) as f:
                crs = f.crs.to_string()
                for geom_idx in geom_idxs:

                    geom = prediction_geoms[geom_idx]
                    geom = fiona.transform.transform_geom("epsg:4326", crs, geom)

                    try:
                        out_image, _ = rasterio.mask.mask(
                            f,
                            [geom],
                            all_touched=True,
                            filled=True,
                            crop=True,
                            pad=True,
                        )
                        out_image = out_image.reshape(4, -1)
                        # which of the locations are nodata
                        mask = (out_image == 0).sum(axis=0) != 4

                        if args.feature_type == "spectral-histograms":
                            for channel_idx in range(4):
                                histograms[geom_idx, channel_idx] = np.bincount(
                                    out_image[channel_idx, mask], minlength=256
                                )
                        elif args.feature_type == "cluster-histograms":
                            out_image = out_image // 64
                            out_image = (out_image.T * BASE_4_ENCODING).sum(axis=1)
                            histograms[geom_idx] = np.bincount(
                                out_image[mask], minlength=256
                            )

                    except Exception:
                        bad_indices.append(geom_idx)

                    if count % 1000 == 0:
                        print(
                            count,
                            len(prediction_geoms),
                            time.time() - tic,
                            count / len(prediction_geoms) * 100,
                        )
                        tic = time.time()
                    count += 1

    # Save output
    np.savez_compressed(args.output_fn, histograms)

    # Save the indices of samples that caused errors
    bad_indices_fn = args.output_fn.replace(".npz", "_bad-indices.txt")
    with open(bad_indices_fn, "w") as f:
        for geom_idx in bad_indices:
            f.write("%d\n" % (geom_idx))


if __name__ == "__main__":
    main()
