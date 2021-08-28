"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.


This script takes a list of GeoTIFFs containing per-pixel poultry barn predictions and
performs the following steps on each:
- Groups sets of contiguous predicted positive pixels into polygons
- For each predicted polygon, computes the following features:
  - The area of the polygon (in square meters)
  - The area of the minimum rotated bounding rectangle covering the polygon (in square
    meters)
  - The aspect ratio of the minimum rotated bounding rectangle covering the polygon
  - The average predicted probability of a positive label over all pixels in the polygon
  - The distance to the nearest road line from OpenStreetMap (in meters)

These _per polygon_ features are then used later in the pipeline to filter out false
positive predictions.

NOTE: The distance to nearest road calculation is performed **only considering roads
that are within the bounds of the GeoTIFF that is being processed**. This can lead to
the case where there exists a road that is closer to a polygon than this script reports.
"""

import argparse
import os
import time

import fiona
import fiona.transform
import networkx as nx
import numpy as np
import osmnx
import pandas as pd
import rasterio
import rasterio.features
import rasterio.mask
import scipy.spatial
import shapely
import shapely.geometry
import shapely.ops
from rasterio.io import MemoryFile

from cafo import utils

parser = argparse.ArgumentParser(description="CAFO result inference script")
# General arguments
parser.add_argument(
    "--input_fn",
    type=str,
    required=True,
    help="The path to a CSV file containing an `image_fn` column.",
)
parser.add_argument(
    "--output_fn", type=str, required=True, help="The output file to write to."
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--blob_root_dir",
    type=str,
    help="The blob container root directory in which to look for files to postprocess"
    + " (we replace the NAIP blob container root with this string when looking for"
    + " output)",
)
group.add_argument(
    "--input_dir",
    type=str,
    help="The flat directory in which to look for the files to postprocess (we will"
    + " replace everything but the input filename with this string when looking for"
    + " output",
)

parser.add_argument(
    "--threshold",
    type=int,
    required=False,
    help="The threshold value [0,255] at which we consider a probabilistic prediction"
    + " to be positive. If this is set then we will look for inputs that end in"
    + " 'predictions.tif', else we will look for inputs that end in"
    + " 'predictions-soft.tif'.",
)
args = parser.parse_args()


def fn_to_date(fn):
    fn = os.path.basename(fn)
    parts = fn.replace(".tif", "").split("_")
    date = parts[-2]
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    return int(year), int(month), int(day)


def postprocess_single_file(
    fn, url, decision_threshold=127, road_step_size=50.0, num_nearest_neighbors=30
):
    """Runs the postprocessing logic on a single GeoTIFF of predictions.

    Args:
        fn: Path to the filename to process -- assumed to have a single channel of
            'uint8' values that represent quantized per pixel probabilities.
        url: Assosciated URL to the image pointed to by `fn`.
        decision_threshold: Threshold value in [0,255] at which a pixel is considered
            a positive prediction.
        road_step_size: The length of the segments that OSM road segments will be
            broken up into.
        num_nearest_neighbors: The number of nearby road segments to consider when
            calculating the distance to the nearest from each polygon.
    """
    # Group contiguous sets of pixels together and calculate geometric features
    predicted_shapes = []
    with rasterio.open(fn) as f:
        src_crs = f.crs.to_string()
        left, bottom, right, top = f.bounds
        if decision_threshold is not None:
            data = f.read()
            mask = (data > decision_threshold).astype(np.uint8)
        else:
            mask = f.read()
        profile = f.profile

        with MemoryFile() as memfile:
            with memfile.open(**profile) as g:
                g.write(mask)
            with memfile.open() as g:
                features = list(
                    rasterio.features.dataset_features(g, 1, geographic=False)
                )

        year, month, day = fn_to_date(fn)

        for j in range(len(features)):
            del features[j]["properties"]
            del features[j]["bbox"]
            shape = shapely.geometry.shape(features[j]["geometry"])
            shape_rectangle = shape.minimum_rotated_rectangle
            predicted_shapes.append(shape_rectangle)
            geom = shapely.geometry.mapping(shape_rectangle)

            out_image, _ = rasterio.mask.mask(f, [geom], crop=True, all_touched=True)

            side_lengths = utils.get_side_lengths(shape_rectangle)
            short_length = min(side_lengths)
            long_length = max(side_lengths)
            aspect_ratio = long_length / short_length

            features[j]["properties"] = {
                "p": out_image.mean() / 255.0,
                "rectangle_area": shape_rectangle.area,
                "area": shape.area,
                "rectangle_aspect_ratio": aspect_ratio,
                "image_url": url,
                "year": year,
                "date": f"{year}-{month}-{day}"
            }

            transformed_geom = fiona.transform.transform_geom(
                src_crs, "epsg:4326", geom
            )
            features[j]["geometry"] = transformed_geom

    if len(features) == 0:
        return []

    # Run distance to nearest road calculations for every polygon we found
    empty = False
    lons, lats = fiona.transform.transform(
        src_crs, "epsg:4326", [left, right], [top, bottom]
    )
    north, south, east, west = lats[0], lats[1], lons[1], lons[0]
    try:
        G = osmnx.graph_from_bbox(
            north,
            south,
            east,
            west,
            network_type="all",
            retain_all=True,
            truncate_by_edge=True,
            clean_periphery=False,
        )
        G = osmnx.project_graph(G, to_crs=src_crs)
    except osmnx.graph.EmptyOverpassResponse:
        empty = True
    except nx.NetworkXPointlessConcept:
        empty = True
    except UnboundLocalError:
        empty = True
    except ValueError:
        empty = True

    if not empty:
        points = []
        road_idxs = []
        roads = []
        edges = set()
        road_idx = 0
        for u, v in G.edges():
            if not ((u, v) in edges or (v, u) in edges):
                edges.add((u, v))
                for edge in G[u][v].values():
                    road = edge["geometry"]
                    roads.append(road)

                    # https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely
                    if road.length < road_step_size:
                        for x, y in zip(*road.xy):
                            points.append((x, y))
                            road_idxs.append(road_idx)
                    else:
                        for d in np.arange(0, road.length, road_step_size):
                            s = shapely.ops.substring(road, d, d + road_step_size)
                            points.append((s.xy[0][0], s.xy[1][0]))
                            road_idxs.append(road_idx)
                    road_idx += 1
        spatial_index = scipy.spatial.cKDTree(points)

        # Calculate distance to nearest road for each predicted shape
        for j in range(len(features)):
            predicted_centroid = (
                predicted_shapes[j].centroid.xy[0][0],
                predicted_shapes[j].centroid.xy[1][0],
            )
            _, idxs = spatial_index.query(predicted_centroid, k=num_nearest_neighbors)

            min_road_distance = float("inf")
            for idx in idxs:
                if idx != len(points):
                    road_idx = road_idxs[idx]
                    t_dist = predicted_shapes[j].distance(roads[road_idx])
                    if t_dist < min_road_distance:
                        min_road_distance = t_dist

            features[j]["properties"]["distance_to_nearest_road"] = min_road_distance
    else:
        for j in range(len(features)):
            features[j]["properties"]["distance_to_nearest_road"] = float("inf")

    return features


def main():

    # Check to make sure input/output files exist/do not exist
    assert os.path.exists(args.input_fn)
    if os.path.exists(args.output_fn):
        print("Output file already exists, exiting...")
        return

    # Read inputs
    df = pd.read_csv(args.input_fn)
    fns = df.image_fn.values
    # Determine what the input files should look like
    if args.threshold is None:
        input_file_pattern = "_predictions.tif"
    else:
        input_file_pattern = "_predictions-soft.tif"

    for fn in fns:
        assert fn.startswith("https://")

    # Calculate the paths to each file that we will be reading
    input_fns = []
    input_urls = []
    for fn in fns:
        input_urls.append(fn)
        if args.blob_root_dir is not None:
            input_fns.append(
                fn.replace(utils.NAIP_BLOB_ROOT, args.blob_root_dir).replace(
                    ".tif", input_file_pattern
                )
            )
        elif args.input_dir is not None:

            input_fn = os.path.basename(fn).replace(".tif", input_file_pattern)
            input_fns.append(os.path.join(args.input_dir, input_fn))
        else:
            raise ValueError(
                "We expect one of --blob_root_dir or --input_dir to be provided"
            )

    # Run postprocessing on all files
    all_features = []
    tic = time.time()
    for i, (fn, url) in enumerate(zip(input_fns, input_urls)):
        if i % 20 == 0:
            print(
                "%d/%d files\t%0.2f seconds\t%d features processed"
                % (i, len(fns), time.time() - tic, len(all_features))
            )
            tic = time.time()
        features = postprocess_single_file(fn, url, decision_threshold=args.threshold)
        for feature in features:
            all_features.append(feature)

    # Write results to an output GeoJSON file
    schema = {
        "properties": {
            "p": "float",
            "rectangle_area": "float",
            "area": "float",
            "rectangle_aspect_ratio": "float",
            "distance_to_nearest_road": "float",
            "year": "int",
            "date": "str",
            "image_url": "str"
        },
        "geometry": "Polygon",
    }
    with fiona.open(
        args.output_fn, "w", driver="GeoJSON", crs="epsg:4326", schema=schema
    ) as f:
        f.writerecords(all_features)


if __name__ == "__main__":
    main()
