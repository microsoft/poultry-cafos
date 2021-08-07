"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import os

import numpy as np
import rasterio
import torch
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from torch.utils.data.dataset import IterableDataset

from ..utils import RASTERIO_BEST_PRACTICES

os.environ.update(RASTERIO_BEST_PRACTICES)


class StreamingGeospatialDataset(IterableDataset):
    def __init__(
        self,
        imagery_fns,
        label_fns=None,
        groups=None,
        chip_size=256,
        num_chips_per_tile=200,
        windowed_sampling=False,
        sample_transform=None,
        nodata_check=None,
        verbose=False,
    ):
        """A torch Dataset for randomly sampling chips from a list of tiles. When used
        in conjunction with a DataLoader that has `num_workers>1` this Dataset will
        assign each worker to sample chips from disjoint sets of tiles.

        Args:
            imagery_fns: A list of filenames (or URLS -- anything that `rasterio.open()`
                can read) pointing to imagery tiles.
            label_fns: A list of filenames of the same size as `imagery_fns` pointing to
                label mask tiles or `None` if the Dataset should operate in "imagery
                only mode". Note that we expect `imagery_fns[i]` and `label_fns[i]` to
                have the same dimension and coordinate system.
            chip_size: Desired size of chips (in pixels).
            num_chips_per_tile: Desired number of chips to sample for each tile.
            windowed_sampling: Flag indicating whether we should sample each chip with a
                read using `rasterio.windows.Window` or whether we should read the whole
                tile into memory, then sample chips.
            sample_transform: A function to apply to each image,label chip. This
                function should return both as a `torch.Tensor`.
            nodata_check: A method that will check an `(image_chip)` or
                `(image_chip, label_chip)` (if `label_fns` are provided) and return
                whether or not the chip should be skipped. This can be used, for
                example, to skip chips that contain nodata values.
            verbose: If `False` we will be quiet.
        """

        self.fns = list(zip(imagery_fns, label_fns))

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling = windowed_sampling

        self.sample_transform = sample_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if (
            worker_info is None
        ):  # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # We only want to shuffle the order we traverse the files if we are the first
        # worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random.shuffle(self.fns)  # in place
        # NOTE: A warning, when different workers are created they will all have the
        # same numpy random seed, however will have different torch random seeds.
        # If you want to use numpy random functions, seed appropriately. E.g:
        # seed = torch.randint(low=0,high=2**32-1,size=(1,)).item()
        # np.random.seed(seed)

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each
        # worker will recieve ceil(num_filenames / num_workers) filenames to generate
        # chips from. If the number of workers doesn't divide the number of filenames
        # evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id + 1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            img_fn, label_fn = self.fns[idx]

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, label_fn)

    def stream_chips(self):
        for img_fn, label_fn in self.stream_tile_fns():

            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            label_fp = rasterio.open(label_fn, "r")

            height, width = img_fp.shape
            t_height, t_width = label_fp.shape
            assert height == t_height and width == t_width

            try:
                # If we aren't in windowed sampling mode then we should read the entire
                # tile up front
                if not self.windowed_sampling:
                    img_data = np.rollaxis(img_fp.read(), 0, 3)
                    label_data = (
                        label_fp.read().squeeze()
                    )  # assume the label geotiff has a single channel

                for i in range(self.num_chips_per_tile):
                    # Select the top left pixel of our chip randomly
                    x = np.random.randint(0, width - self.chip_size)
                    y = np.random.randint(0, height - self.chip_size)

                    # Read imagery / labels
                    img = None
                    labels = None
                    if self.windowed_sampling:
                        img = np.rollaxis(
                            img_fp.read(
                                window=Window(x, y, self.chip_size, self.chip_size)
                            ),
                            0,
                            3,
                        )
                        labels = label_fp.read(
                            window=Window(x, y, self.chip_size, self.chip_size)
                        ).squeeze()
                    else:
                        img = img_data[
                            y : y + self.chip_size, x : x + self.chip_size, :
                        ]
                        labels = label_data[
                            y : y + self.chip_size, x : x + self.chip_size
                        ]

                    # Check for no data
                    if self.nodata_check is not None:
                        skip_chip = self.nodata_check(img, labels)

                        if skip_chip:  # The current chip has been identified as invalid
                            num_skipped_chips += 1
                            continue

                    img, labels = self.sample_transform(img, labels)

                    # Note, that img should be a torch "Double" type (i.e. a np.float32)
                    # and labels should be a torch "Long" type (i.e. np.int64)
                    yield img, labels
            except RasterioIOError:
                print("WARNING: Reading %s failed, skipping..." % (img_fn))

            # Close file pointers
            img_fp.close()
            label_fp.close()

            if num_skipped_chips > 0 and self.verbose:
                print("We skipped %d chips on %s" % (num_skipped_chips, img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())
