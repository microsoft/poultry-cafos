"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Version of the inference script that writes all output to a single directory.
"""
import argparse
import datetime
import os
import time

import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import torch
import torch.nn.functional as F

from cafo import models, utils
from cafo.data.TileDatasets import TileInferenceDataset

os.environ.update(utils.RASTERIO_BEST_PRACTICES)

NUM_WORKERS = 4
CHIP_SIZE = 256
PADDING = 32
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING

parser = argparse.ArgumentParser(description="CAFO detection model inference script")

parser.add_argument(
    "--input_fn",
    type=str,
    required=True,
    help="Path to a text file containing a list of files to run the model on.",
)
parser.add_argument(
    "--model_fn", type=str, required=True, help="Path to the model file to use."
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to a directory where outputs will be saved. This directory will be"
    + " created if it does not exist.",
)
parser.add_argument("--gpu", type=int, default=0, help="ID of the GPU to run on.")
parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size to use during inference."
)

parser.add_argument(
    "--model",
    default="unet",
    choices=("unet", "manet", "unet++", "deeplabv3+"),
    help="Model to use",
)

parser.add_argument(
    "--save_soft", action="store_true", help="Whether to save soft versions of output."
)

args = parser.parse_args()


def main():
    print(
        "Starting CAFO detection model inference script at %s"
        % (str(datetime.datetime.now()))
    )

    # Load files
    assert os.path.exists(args.input_fn)
    assert os.path.exists(args.model_fn)

    # Ensure output directory exists
    if os.path.exists(args.output_dir):
        if len(os.listdir(args.output_dir)) > 0:
            print(
                "WARNING: The output directory is not empty, but we are ignoring that"
                + " and writing data."
            )
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    print("Running on %d files" % (len(image_fns)))

    # Load model
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
    else:
        print("WARNING! Torch is reporting that CUDA isn't available, exiting...")
        return
    print("Using device:", device)

    if args.model == "unet":
        model = models.get_unet()
    elif args.model == "unet++":
        model = models.get_fcn()
    elif args.model == "manet":
        model = models.get_manet()
    elif args.model == "deeplabv3+":
        model = models.get_deeplab()
    else:
        raise ValueError("Invalid model")
    model.load_state_dict(
        torch.load(args.model_fn, map_location="cpu")["model_checkpoint"]
    )
    model = model.to(device)

    # Run model on all files and save output
    for image_idx, image_fn in enumerate(image_fns):
        tic = time.time()

        with rasterio.open(image_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        dataset = TileInferenceDataset(
            image_fn,
            chip_size=CHIP_SIZE,
            stride=CHIP_STRIDE,
            transform=utils.chip_transformer,
            verbose=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Run model and organize output
        output = np.zeros((2, input_height, input_width), dtype=np.float32)
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, coords) in enumerate(dataloader):
            data = data.to(device)
            with torch.no_grad():
                t_output = model(data)
                t_output = F.softmax(t_output, dim=1).cpu().numpy()

            for j in range(t_output.shape[0]):
                y, x = coords[j]

                output[:, y : y + CHIP_SIZE, x : x + CHIP_SIZE] += t_output[j] * kernel
                counts[y : y + CHIP_SIZE, x : x + CHIP_SIZE] += kernel

        output = output / counts

        # Save output
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["compress"] = "lzw"
        output_profile["predictor"] = 2
        output_profile["count"] = 1
        output_profile["nodata"] = 0
        output_profile["tiled"] = True
        output_profile["blockxsize"] = 512
        output_profile["blockysize"] = 512

        if args.save_soft:
            output = output / output.sum(axis=0, keepdims=True)
            output = (output * 255).astype(np.uint8)

            output_fn = image_fn.split("/")[-1].replace(".tif", "_predictions-soft.tif")
            output_fn = os.path.join(args.output_dir, output_fn)

            with rasterio.open(output_fn, "w", **output_profile) as f:
                f.write(output[1], 1)

        else:
            output_hard = output.argmax(axis=0).astype(np.uint8)

            output_fn = image_fn.split("/")[-1].replace(".tif", "_predictions.tif")
            output_fn = os.path.join(args.output_dir, output_fn)

            with rasterio.open(output_fn, "w", **output_profile) as f:
                f.write(output_hard, 1)
                f.write_colormap(
                    1,
                    {
                        0: (0, 0, 0, 0),
                        1: (255, 0, 0, 255),
                    },
                )

        print(
            "(%d/%d) Processed %s in %0.4f seconds"
            % (image_idx, len(image_fns), image_fn, time.time() - tic)
        )


if __name__ == "__main__":
    main()
