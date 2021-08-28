"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs the best model checkpoints over the test set and saves the predictions.
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
    output_dir = f"output/inference/{run}/"

    if not os.path.exists(output_dir):
        commands.append(
            f"python inference.py --input_fn {input_fn}"
            + f" --model_fn {model_fn} --output_dir {output_dir} --gpu {GPU}"
        )
    else:
        print(f"{output_dir} already exists, skipping!")

for i, command in enumerate(commands):
    print(f"{i}/{len(commands)}")
    subprocess.call(command, shell=True)
