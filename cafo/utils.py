"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from parse import parse

# Some tricks to make rasterio faster when using vsicurl
# see https://github.com/pangeo-data/cog-best-practices
RASTERIO_BEST_PRACTICES = dict(
    CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_MAX_RAW_BLOCK_CACHE_SIZE="200000000",
    GDAL_SWATH_SIZE="200000000",
    VSI_CURL_CACHE_SIZE="200000000",
)

NAIP_BLOB_ROOT = "https://naipblobs.blob.core.windows.net/naip"
LOGGER = logging.getLogger("main")


# Filter method
def filter_polygon(**kwargs):
    """Decides whether a predicted polygon is valid based on the range of the aspect
    ratio and area stats from the Delmarava dataset.
    """
    return all(
        [
            (
                kwargs["distance_to_nearest_road"] is None
                or kwargs["distance_to_nearest_road"] > 0
            ),
            kwargs["rectangle_aspect_ratio"] > 3.4,
            kwargs["rectangle_aspect_ratio"] < 20.49,
            kwargs["rectangle_area"] > 525.69,
            kwargs["rectangle_area"] < 8106.53,
        ]
    )


# Helper function for parsing results
def convert_results_to_series(results):
    series = {key: [] for key in results[0].keys()}
    for epoch in results:
        for k, v in epoch.items():
            series[k].append(v)
    return series


def parse_fn_parts(run):
    parsed_run = None
    if "_rotation" in run:
        parsed_run = parse(
            "{training_set}_{model}_{negative_sample_probability:f}_{lr:f}_rotation",
            run,
        ).named
        parsed_run["rotation"] = True
    else:
        parsed_run = parse(
            "{training_set}_{model}_{negative_sample_probability:f}_{lr:f}", run
        ).named
        parsed_run["rotation"] = False
    return parsed_run


# Method for creating torch tensor from input chip
def chip_transformer(img):
    img = img / 255.0
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img


# Training methods
def fit(model, device, data_loader, num_batches, optimizer, criterion, epoch, memo=""):
    model.train()

    losses = []
    tic = time.time()
    for batch_idx, (data, targets) in enumerate(data_loader):
        if batch_idx % 10 == 0:
            elapsed_time = time.time() - tic
            remaining_time = (
                (elapsed_time / (batch_idx + 1)) * num_batches
            ) - elapsed_time
            LOGGER.info(
                f"Epoch {epoch} -- Batch {batch_idx}/{num_batches} --"
                + f" {elapsed_time / 60:0.4f} minutes elapsed --"
                + f" {(batch_idx + 1) / elapsed_time:0.4f} batches/second --"
                + f" estimated {remaining_time / 60:0.4f} minutes remaining"
            )

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)

    LOGGER.info(
        "[{}] Training Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}".format(
            memo, epoch, time.time() - tic, avg_loss
        )
    )

    return [avg_loss]


def evaluate(model, device, data_loader, num_batches, criterion, epoch, memo=""):
    model.eval()

    losses = []
    tic = time.time()
    for batch_idx, (data, targets) in enumerate(data_loader):
        if batch_idx % 1000 == 0:
            elapsed_time = time.time() - tic
            remaining_time = (
                (elapsed_time / (batch_idx + 1)) * num_batches
            ) - elapsed_time
            LOGGER.info(
                "%d/%d -- %0.4f minutes elapsed -- estimated %0.4f minutes remaining"
                % (batch_idx, num_batches, elapsed_time / 60, remaining_time / 60)
            )

        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

    avg_loss = np.mean(losses)

    LOGGER.info(
        "[{}] Validation Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}".format(
            memo, epoch, time.time() - tic, avg_loss
        )
    )

    return [avg_loss]


def score(model, device, data_loader, num_batches):
    model.eval()

    num_classes = model.module.segmentation_head[0].out_channels
    num_samples = len(data_loader.dataset)
    predictions = np.zeros((num_samples, num_classes), dtype=np.float32)
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            output = F.softmax(model(data))
        batch_size = data.shape[0]
        predictions[idx : idx + batch_size] = output.cpu().numpy()
        idx += batch_size
    return predictions


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Misc methods
def setup_log_file_handler(save_dir, suffix="scratch"):
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)

    printFormatter = logging.Formatter("[%(asctime)s] - %(message)s")
    fileFormatter = logging.Formatter(
        "%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s"
    )

    log_strHandler = logging.StreamHandler()
    log_strHandler.setFormatter(printFormatter)
    logger.addHandler(log_strHandler)

    log_filename = os.path.join(save_dir, "logs_{}.txt".format(suffix))
    log_fileHandler = logging.FileHandler(log_filename)
    log_fileHandler.setFormatter(fileFormatter)
    logger.addHandler(log_fileHandler)

    return logger


# Geometry methods
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def get_side_lengths(shape):
    xs, ys = shape.boundary.xy
    pts = list(zip(xs, ys))
    lengths = []
    for i in range(len(pts) - 1):
        lengths.append(distance(pts[i], pts[i + 1]))
    assert len(lengths) == 4
    return sorted(lengths)


def reverse_polygon(polygon):
    new_coords = []
    if len(polygon[0][0]) == 2:
        for ring in polygon:
            new_ring = []
            for x, y in ring:
                new_ring.append((y, x))
            new_coords.append(new_ring)
    else:
        for ring in polygon:
            new_ring = []
            for x, y, z in ring:
                new_ring.append((y, x))
            new_coords.append(new_ring)
    return new_coords


def reverse_polygon_coordinates(geom):
    if geom["type"] == "MultiPolygon":
        new_coords = []
        for polygon in geom["coordinates"]:
            new_polygon = reverse_polygon(polygon)
            new_coords.append(new_polygon)
    elif geom["type"] == "Polygon":
        new_coords = reverse_polygon(geom["coordinates"])
    geom["coordinates"] = new_coords
    return geom
