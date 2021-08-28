"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import argparse
import copy
import datetime
import os
import time

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.transform import rotate

from cafo import models, utils
from cafo.data.StreamingDatasets import StreamingGeospatialDataset
from cafo.data.TileDatasets import TileInferenceDataset

os.environ.update(utils.RASTERIO_BEST_PRACTICES)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description="CAFO model training script")
# General arguments
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="The path to a directory to store model checkpoints",
)
parser.add_argument(
    "--data_blob_root",
    type=str,
    required=False,
    help="The prefix to append to the paths of the tiles that we are loading",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Flag for overwriting `output_dir` if that directory already exists",
)
parser.add_argument(
    "--save_most_recent",
    action="store_true",
    help="Flag for saving the most recent version of the model during training",
)
parser.add_argument(
    "--azureml",
    action="store_true",
    help="Whether we are running experiments on Azure ML",
)
parser.add_argument("--gpu", type=int, default=-1, help="The ID of the GPU to use")
parser.add_argument(
    "--debug",
    action="store_true",
    help="This drops all but a few tiles so we can test everything",
)

# Dataloader
parser.add_argument(
    "--num_dataloader_workers",
    type=int,
    default=6,
    help="Number of workers to use in all dataloaders",
)
parser.add_argument(
    "--num_chips_per_tile",
    type=int,
    default=600,
    help="The number of chips we will sample from each tile (we will potentially reject"
    + " some of these, so this isn't fixed)",
)
parser.add_argument(
    "--chip_size",
    type=int,
    default=256,
    help="The size of each chip to pass to the model",
)
parser.add_argument(
    "--inference_padding",
    type=int,
    default=32,
    help="The amount to padding to throw away from each chip during inference (must be"
    + " an even number)",
)

# Experiment arguments
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed to pass to numpy and torch"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size to use for training"
)
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
parser.add_argument(
    "--num_epochs", type=int, default=50, help="Number of epochs to train for"
)
parser.add_argument(
    "--rotation_augmentation",
    action="store_true",
    help="Whether to use rotation augmentation",
)
parser.add_argument(
    "--negative_sample_probability",
    type=float,
    default=1.0,
    help="Probability that we will sample a chip given that it doesn't have some of the"
    + " positive class (we will always sample if there is some positive class)",
)
parser.add_argument(
    "--model",
    default="unet",
    choices=("unet", "unet-large", "fcn"),
    help="Model to use",
)
parser.add_argument(
    "--training_set",
    default="train-all",
    choices=("train-all", "train-single", "train-augment", "all-all", "all-augment"),
    help="Which training set to use",
)

args = parser.parse_args()


NUM_WORKERS = args.num_dataloader_workers
NUM_CHIPS_PER_TILE = args.num_chips_per_tile
CHIP_SIZE = args.chip_size
LARGE_CHIP_SIZE = int(np.ceil(CHIP_SIZE * np.sqrt(2)))
CROP_POINT = (LARGE_CHIP_SIZE - CHIP_SIZE) // 2

PADDING = args.inference_padding
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING


def joint_transform(img, labels):
    if args.rotation_augmentation:
        rotate_amount = np.random.randint(0, 360)
        img = rotate(img, rotate_amount)
        labels = rotate(labels, rotate_amount, order=0)
        labels = (labels * 255).astype(np.uint8)

        img = img[
            CROP_POINT : CROP_POINT + CHIP_SIZE, CROP_POINT : CROP_POINT + CHIP_SIZE
        ]
        labels = labels[
            CROP_POINT : CROP_POINT + CHIP_SIZE, CROP_POINT : CROP_POINT + CHIP_SIZE
        ]
    else:
        img = img / 255.0

        img = img[
            CROP_POINT : CROP_POINT + CHIP_SIZE, CROP_POINT : CROP_POINT + CHIP_SIZE
        ]
        labels = labels[
            CROP_POINT : CROP_POINT + CHIP_SIZE, CROP_POINT : CROP_POINT + CHIP_SIZE
        ]

    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    labels = labels.astype(np.int64)
    labels = torch.from_numpy(labels)

    return img, labels


def skip_check(img, labels):
    if np.any(
        np.sum(img == 0, axis=2) == 4
    ):  # if we have an all black part of NAIP then skip
        return True
    elif np.any(labels == 1):  # else, if we have any positive labels, then don't skip
        return False
    else:  # else, skip with probability `negative_sample_probability`
        return np.random.random() >= args.negative_sample_probability


def do_validation(
    validation_image_fns, validation_label_fns, model, device, epoch, logger, memo=""
):
    model.eval()

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tn = 0

    y_trues = []
    y_preds = []

    per_tile_ious = []
    per_tile_recalls = []
    per_tile_precisions = []

    tic = time.time()
    for validation_image_fn, validation_label_fn in zip(
        validation_image_fns, validation_label_fns
    ):

        val_dataset = TileInferenceDataset(
            validation_image_fn,
            chip_size=CHIP_SIZE,
            stride=CHIP_STRIDE,
            transform=utils.chip_transformer,
            verbose=False,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        with rasterio.open(validation_label_fn) as f:
            y_true = f.read().squeeze()
            input_height, input_width = y_true.shape

        output = np.zeros((2, input_height, input_width), dtype=np.float32)
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, coords) in enumerate(val_dataloader):
            data = data.to(device)
            with torch.no_grad():
                t_output = model(data)
                t_output = F.softmax(t_output, dim=1).cpu().numpy()

            for j in range(t_output.shape[0]):
                y, x = coords[j]

                output[:, y : y + CHIP_SIZE, x : x + CHIP_SIZE] += t_output[j] * kernel
                counts[y : y + CHIP_SIZE, x : x + CHIP_SIZE] += kernel

        output = output / counts
        y_pred = output.argmax(axis=0).astype(np.uint8)

        gt_positives = y_true == 1
        gt_negatives = y_true == 0
        pred_positives = y_pred == 1
        pred_negatives = y_pred == 0

        tp = np.sum(gt_positives & pred_positives)
        fp = np.sum(gt_negatives & pred_positives)
        fn = np.sum(gt_positives & pred_negatives)
        tn = np.sum(gt_negatives & pred_negatives)

        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_tn += tn

        # Record a sample of pixels to compute more expensive metrics
        y_trues.append(y_true.ravel()[::100])
        y_preds.append(output[1].ravel()[::100])

        iou = tp / (tp + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        per_tile_ious.append(iou)
        per_tile_recalls.append(recall)
        per_tile_precisions.append(precision)

    iou = all_tp / (all_tp + all_fp + all_fn)
    recall = all_tp / (all_tp + all_fn)
    precision = all_tp / (all_tp + all_fp)

    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)

    logger.info(
        "[{}] Validation Epoch: {}\t Time elapsed: {:.2f} seconds".format(
            memo, epoch, time.time() - tic
        )
    )
    logger.info("\tIoU: {}".format(iou))
    logger.info("\tPrecision: {}".format(precision))
    logger.info("\tRecall: {}".format(recall))

    return {
        "val_iou": iou,
        "val_recall": recall if not np.isnan(precision) else -1,
        "val_precision": precision if not np.isnan(precision) else -1,
        "per_tile_ious": per_tile_ious,
        "per_tile_recalls": per_tile_recalls,
        "per_tile_precisions": per_tile_precisions,
    }


def main():

    # Setup
    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)):
        if args.overwrite:
            print(
                f"WARNING! The output directory, {args.output_dir}, already exists, we"
                + " might overwrite data in it!" % (args.output_dir)
            )
        else:
            print(
                f"The output directory, {args.output_dir}, already exists and isn't"
                + "empty. We don't want to overwrite and existing results, exiting..."
            )
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if args.azureml:
        from azureml.core import Run
        run = Run.get_context()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%X")
    logger = utils.setup_log_file_handler(
        args.output_dir, "training_{}".format(now_str)
    )
    logger.info("Starting CAFO training script")
    logger.info("Saving results to: {}".format(args.output_dir))

    if torch.cuda.is_available():
        if args.gpu == -1:
            device = torch.device("cuda")
            logger.info("Using %d devices" % (torch.cuda.device_count()))
        else:
            device = torch.device("cuda:%d" % args.gpu)
            logger.info("Using a single device")
    else:
        logger.error(
            "WARNING! Torch is reporting that CUDA isn't available, exiting..."
        )
        return
    logger.info("Using device: %s" % (str(device)))

    # Load input data
    validation_image_fns = [
        "naip/v002/de/2011/de_100cm_2011/38075/m_3807505_ne_18_1_20110602.tif",
        "naip/v002/de/2013/de_100cm_2013/38075/m_3807505_ne_18_1_20130915.tif",
        "naip/v002/de/2015/de_100cm_2015/38075/m_3807505_ne_18_1_20150629.tif",
        "naip/v002/de/2017/de_100cm_2017/38075/m_3807505_ne_18_1_20170720.tif",
        "naip/v002/de/2018/de_060cm_2018/38075/m_3807505_ne_18_060_20180827.tif",
    ]
    validation_label_fns = [
        "train-augment/v002/de/2011/de_100cm_2011/38075/m_3807505_ne_18_1_20110602.tif",
        "train-augment/v002/de/2013/de_100cm_2013/38075/m_3807505_ne_18_1_20130915.tif",
        "train-augment/v002/de/2015/de_100cm_2015/38075/m_3807505_ne_18_1_20150629.tif",
        "train-augment/v002/de/2017/de_100cm_2017/38075/m_3807505_ne_18_1_20170720.tif",
        "train-augment/v002/de/2018/de_060cm_2018/38075/m_3807505_ne_18_060_20180827.tif",
    ]

    if args.training_set == "train-all":
        input_fn = "data/splits/train-all.csv"
    elif args.training_set == "train-single":
        input_fn = "data/splits/train-single.csv"
    elif args.training_set == "train-augment":
        input_fn = "data/splits/train-augment.csv"
    elif args.training_set == "all-all":
        input_fn = "data/splits/all.csv"

    input_dataframe = pd.read_csv(input_fn)
    image_fns = input_dataframe["image_fn"].to_list()
    label_fns = input_dataframe["label_fn"].to_list()

    if args.debug:
        image_fns = image_fns[:4]
        label_fns = label_fns[:4]

    # remove val tile from training set
    image_fns = [fn for fn in image_fns if "m_3807505_ne_18" not in fn]
    label_fns = [fn for fn in label_fns if "m_3807505_ne_18" not in fn]

    if args.data_blob_root is not None:
        image_fns = [args.data_blob_root + fn for fn in image_fns]
        label_fns = [args.data_blob_root + fn for fn in label_fns]

        validation_image_fns = [args.data_blob_root + fn for fn in validation_image_fns]
        validation_label_fns = [args.data_blob_root + fn for fn in validation_label_fns]

    image_fns = np.array(image_fns)
    label_fns = np.array(label_fns)

    train_dataset = StreamingGeospatialDataset(
        imagery_fns=image_fns,
        label_fns=label_fns,
        chip_size=LARGE_CHIP_SIZE,
        num_chips_per_tile=NUM_CHIPS_PER_TILE,
        windowed_sampling=False,
        verbose=False,
        sample_transform=joint_transform,
        nodata_check=skip_check,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_training_batches_per_epoch = int(
        len(image_fns) * NUM_CHIPS_PER_TILE / args.batch_size
    )
    logger.info("We will be training with %d different tiles" % (image_fns.shape[0]))
    logger.info(
        "We will be training with %d batches per epoch"
        % (num_training_batches_per_epoch)
    )

    # Setup training
    if args.model == "unet":
        model = models.get_unet()
    elif args.model == "fcn":
        model = models.get_fcn()
    elif args.model == "unet-large":
        model = models.get_unet_large()
    else:
        raise ValueError("Invalid model")

    if args.gpu == -1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, threshold=0.0001
    )

    logger.info("Model has %d parameters" % (utils.count_parameters(model)))

    # Model training
    metrics_per_epoch = []
    best_val_iou = 0
    num_times_lr_dropped = 0

    for epoch in range(args.num_epochs):
        lr = utils.get_lr(optimizer)

        training_losses = utils.fit(
            model,
            device,
            train_dataloader,
            num_training_batches_per_epoch,
            optimizer,
            criterion,
            epoch,
        )
        metrics = do_validation(
            validation_image_fns,
            validation_label_fns,
            model,
            device,
            epoch,
            logger,
            memo="",
        )

        # Record training loss and val metrics
        metrics["training_loss"] = training_losses[0]
        metrics_per_epoch.append(metrics)

        if args.azureml:
            run.log("training_loss", metrics["training_loss"])
            run.log("val_iou", metrics["val_iou"])
            run.log("val_precision", metrics["val_precision"])
            run.log("val_recall", metrics["val_recall"])
            run.log("epoch", epoch)

        # LR schedule / early stopping
        scheduler.step(training_losses[0])
        if utils.get_lr(optimizer) < lr:
            num_times_lr_dropped += 1
            logger.info("")
            logger.info("Learning rate dropped")
            logger.info("")

        # Save everything
        save_obj = {
            "epoch": epoch,
            "optimizer_checkpoint": copy.deepcopy(optimizer.state_dict()),
            "model_checkpoint": copy.deepcopy(model.state_dict()),
        }
        torch.save(
            save_obj, os.path.join(args.output_dir, "checkpoint_epoch_%d.pt" % (epoch))
        )
        if metrics["val_iou"] > best_val_iou:
            logger.info("New best!")
            best_val_iou = metrics["val_iou"]
            torch.save(save_obj, os.path.join(args.output_dir, "best_checkpoint.pt"))

        torch.save(
            {"metrics_per_epoch": metrics_per_epoch, "args": args},
            os.path.join(args.output_dir, "results.pt"),
        )

        if num_times_lr_dropped == 4:
            break

    # Cleanup
    logger.info("Finished training run")


if __name__ == "__main__":
    main()
