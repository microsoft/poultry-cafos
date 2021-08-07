"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs the best model checkpoints over the test set and records performance.
"""
import os
import subprocess

BASE_DIR = "outputs/training/"
GPU = 0

runs = os.listdir(BASE_DIR)
commands = []
for run in runs:
    model_fn = os.path.join(BASE_DIR, run, "best_checkpoint.pt")
    input_fn = "data/splits/test-single.csv"
    output_fn = f"output/evaluation/{run}.csv"

    commands.append(
        f"python inference_and_evaluate.py --input_fn {input_fn}"
        + f" --model_fn {model_fn} --output_fn {output_fn} --gpu {GPU}"
    )

for i, command in enumerate(commands):
    print("%d/%d -- %s" % (i, len(commands), commands))
    subprocess.call(command, shell=True)
