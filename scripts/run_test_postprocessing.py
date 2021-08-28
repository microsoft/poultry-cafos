"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs postprocessing over model predictions on the test set.
"""
import os
import subprocess

BASE_DIR = "output/training/"
GPU = 0

runs = os.listdir(BASE_DIR)
commands = []
for run in runs:
    model_fn = os.path.join(BASE_DIR, run, "best_checkpoint.pt")
    input_fn = "data/splits/test-single.csv"
    output_fn = f"output/inference/{run}.geojson"
    input_dir = f"output/inference/{run}/"

    if not os.path.exists(output_fn):
        commands.append(
            f"python postprocess.py --input_fn {input_fn} --output_fn {output_fn}"
            + f" --input_dir {input_dir}"
        )
    else:
        print(f"{output_fn} already exists, skipping!")

for i, command in enumerate(commands):
    print(f"{i}/{len(commands)}")
    subprocess.call(command, shell=True)
