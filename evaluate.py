"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import argparse
import datetime
import os
import time

import numpy as np
import pandas as pd
import rasterio

from cafo.utils import RASTERIO_BEST_PRACTICES

os.environ.update(RASTERIO_BEST_PRACTICES)


parser = argparse.ArgumentParser(description="CAFO test set evaluation script")

parser.add_argument(
    "--input_fn",
    type=str,
    required=True,
    help="Path to a text file containing a list of label files that we expect to"
    + "evaluate against.",
)
parser.add_argument(
    "--predictions_dir",
    type=str,
    required=True,
    help="Path to the directory that contains the predictions.",
)
parser.add_argument(
    "--output_fn",
    type=str,
    required=True,
    help="Path to the file that we want to save the output in.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Flag for overwriting `output_fn` if that directory already exists",
)
args = parser.parse_args()


def main():
    print(
        "Starting CAFO test set evaluation script at %s"
        % (str(datetime.datetime.now()))
    )

    # Load files
    assert os.path.exists(args.input_fn)
    assert os.path.exists(args.predictions_dir)

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
    label_fns = input_dataframe["label_fn"].values
    print("Evaluating on %d files" % (len(label_fns)))

    prediction_fns = []
    for label_fn in label_fns:
        new_fn = label_fn.split("/")[-1].replace(".tif", "_predictions.tif")
        prediction_fn = os.path.join(args.predictions_dir, new_fn)
        assert os.path.isfile(prediction_fn)
        prediction_fns.append(prediction_fn)

    # Run model on all files and save output
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tn = 0

    with open(args.output_fn, "w") as results_f:
        results_f.write("label_fn,prediction_fn,tp,fp,fn,tn,iou,recall,precision,acc\n")
        for image_idx, (prediction_fn, label_fn) in enumerate(
            zip(prediction_fns, label_fns)
        ):
            tic = time.time()

            print("(%d/%d)" % (image_idx, len(label_fns)), end=" ... ")

            with rasterio.open(label_fn) as f:
                y_true = f.read().squeeze()

            with rasterio.open(prediction_fn) as f:
                y_pred = f.read().squeeze()

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
            acc = (tp + tn) / (tp + tn + fp + fn)

            all_tp += int(tp)
            all_fp += int(fp)
            all_fn += int(fn)
            all_tn += int(tn)

            print("finished in %0.4f seconds" % (time.time() - tic))

            results_f.write(
                f"{label_fn},{prediction_fn},{tp},{fp},{fn},{tn},{iou},{recall},"
                + f"{precision},{acc}\n"
            )

        all_iou = all_tp / (all_tp + all_fp + all_fn)
        all_recall = all_tp / (all_tp + all_fn)
        all_precision = all_tp / (all_tp + all_fp)
        all_acc = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn)

        results_f.write("----\n")
        results_f.write(
            f",Totals,{all_tp},{all_fp},{all_fn},{all_tn},{all_iou},{all_recall},"
            + f"{all_precision},{all_acc}"
        )

    # Cleanup
    print("IoU: %0.6f" % (all_iou))
    print("Recall: %0.6f" % (all_recall))
    print("Precision: %0.6f" % (all_precision))
    print("ACC: %0.6f" % (all_acc))


if __name__ == "__main__":
    main()
