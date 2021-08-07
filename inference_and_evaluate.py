"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Version of the inference script that also computes evaluation metrics. If you don't want
the intermediate results, then this is more effecient.
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
PADDING = 64
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING

parser = argparse.ArgumentParser(
    description="CAFO model inference and evaluation script"
)

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
    "--output_fn",
    type=str,
    required=True,
    help="Path to the file that we want to save the output in.",
)
parser.add_argument("--gpu", type=int, default=0, help="ID of the GPU to run on.")
parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size to use during inference."
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Flag for overwriting `output_fn` if that directory already exists",
)

parser.add_argument(
    "--model", default="unet", choices=("unet", "fcn"), help="Model to use"
)

args = parser.parse_args()


def main():
    print(
        "Starting CAFO inference and evaluation model inference script at %s"
        % (str(datetime.datetime.now()))
    )

    # Load files
    assert os.path.exists(args.input_fn)
    assert os.path.exists(args.model_fn)

    os.makedirs(os.path.dirname(args.output_fn), exist_ok=True)
    if os.path.exists(args.output_fn):
        if args.overwrite:
            print("WARNING: we are overwriting existing file: %s" % (args.output_fn))
        else:
            print(
                "WARNING: %s already exists and we aren't overwriting, exiting..."
                % (args.output_fn)
            )
            return

    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
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
    elif args.model == "fcn":
        model = models.get_fcn()
    else:
        raise ValueError("Invalid model")
    model.load_state_dict(torch.load(args.model_fn)["model_checkpoint"])
    model = model.to(device)

    # Run model on all files and save output
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tn = 0

    y_trues = []
    y_preds = []

    with open(args.output_fn, "w") as results_f:
        results_f.write("image_fn,label_fn,tp,fp,fn,tn,iou,recall,precision\n")
        for image_idx, (image_fn, label_fn) in enumerate(zip(image_fns, label_fns)):
            tic = time.time()

            print(
                "(%d/%d) Processing %s" % (image_idx, len(image_fns), image_fn),
                end=" ... ",
            )

            with rasterio.open(image_fn) as f:
                input_width, input_height = f.width, f.height

            with rasterio.open(label_fn) as f:
                y_true = f.read().squeeze()

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

                    output[:, y : y + CHIP_SIZE, x : x + CHIP_SIZE] += (
                        t_output[j] * kernel
                    )
                    counts[y : y + CHIP_SIZE, x : x + CHIP_SIZE] += kernel

            output = output / counts
            y_pred = output.argmax(axis=0).astype(np.uint8)

            # get tile results
            gt_positives = y_true == 1
            gt_negatives = y_true == 0
            pred_positives = y_pred == 1
            pred_negatives = y_pred == 0

            tp = np.sum(gt_positives & pred_positives)
            fp = np.sum(gt_negatives & pred_positives)
            fn = np.sum(gt_positives & pred_negatives)
            tn = np.sum(gt_negatives & pred_negatives)

            iou = tp / (tp + fp + fn)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)

            all_tp += int(tp)
            all_fp += int(fp)
            all_fn += int(fn)
            all_tn += int(tn)

            y_trues.append(y_true.ravel()[::100])
            y_preds.append(output[1].ravel()[::100])

            results_f.write(
                f"{image_fn},{label_fn},{tp},{fp},{fn},{tn},{iou},{recall},{precision}\n"
            )
            results_f.flush()
            print("finished in %0.4f seconds" % (time.time() - tic))

        all_iou = all_tp / (all_tp + all_fp + all_fn)
        all_recall = all_tp / (all_tp + all_fn)
        all_precision = all_tp / (all_tp + all_fp)
        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)

        results_f.write("----\n")
        results_f.write(
            f",Totals,{all_tp},{all_fp},{all_fn},{all_tn},{all_iou},{all_recall},"
            + f"{all_precision}"
        )

    # Cleanup
    print("IoU: %0.6f" % (all_iou))
    print("Recall: %0.6f" % (all_recall))
    print("Precision: %0.6f" % (all_precision))


if __name__ == "__main__":
    main()
