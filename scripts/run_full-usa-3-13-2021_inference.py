"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.


Script for running an inference script in parallel over a list of inputs.

We split the actual list of filenames we want to run on into NUM_GPUS different batches,
save those batches to file, and call `inference.py` multiple times in parallel - pointing
it to a different batch each time.
"""
import subprocess
from multiprocessing import Process

import numpy as np

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0, 1, 2, 3]
TEST_MODE = False  # if False then print out the commands to be run, if True then run

# path passed to `--model_fn` in the `inference.py` script
MODEL_FN = "output/train-all_unet_0.5_0.01_rotation_best-checkpoint.pt"
# path passed to `--output_dir` in the `inference.py` script
OUTPUT_DIR = "output/full-usa-3-13-2021/inference/"

# Get the list of files we want our model to run on
with open("data/naip_most_recent_100cm.csv", "r") as f:
    fns = f.read().strip().split("\n")[1:]

# Split the list of files up into approximately equal sized batches based on the number
# of GPUs we want to use. Each worker will then work on NUM_FILES / NUM_GPUS files in
# parallel. Save these batches of the original list to disk (as a simple list of files
# to be consumed by the `inference.py` script)
num_files = len(fns)
num_splits = len(GPUS)
num_files_per_split = np.ceil(num_files / num_splits)

output_fns = []
for split_idx in range(num_splits):
    output_fn = "data/runs/full-usa-3-13-2021_split_%d.csv" % (split_idx)
    with open(output_fn, "w") as f:
        start_range = int(split_idx * num_files_per_split)
        end_range = min(num_files, int((split_idx + 1) * num_files_per_split))
        print("Split %d: %d files" % (split_idx + 1, end_range - start_range))
        f.write("image_fn\n")
        for i in range(start_range, end_range):
            end = "" if i == end_range - 1 else "\n"
            f.write("%s%s" % (fns[i], end))
    output_fns.append(output_fn)


# Start NUM_GPUS worker processes, each pointed to one of the lists of files we saved
# to disk in the previous step.
def do_work(fn, gpu_idx):
    command = f"python inference_large.py --input_fn {fn} --model_fn {MODEL_FN}"
    +f" --base_output_dir {OUTPUT_DIR} --gpu {gpu_idx} --save_soft"
    print(command)
    if not TEST_MODE:
        subprocess.call(command, shell=True)


processes = []
for work, gpu_idx in zip(output_fns, GPUS):
    p = Process(target=do_work, args=(work, gpu_idx))
    processes.append(p)
    p.start()
for p in processes:
    p.join()
