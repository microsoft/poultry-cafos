"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Runs the train script with a grid of hyperparameters.
"""
import itertools
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0, 1, 2, 3]
TEST_MODE = True  # if False then print out the commands to be run, if True then run

# Hyperparameter options
training_set_options = ["train-all", "train-single"]
model_options = ["unet"]
negative_sample_probability_options = [0.05, 0.1, 0.5]
rotation_augmentation_options = ["--rotation_augmentation", ""]
lr_options = [0.01, 0.001]


def do_work(work, gpu_idx):
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(experiment)
        if not TEST_MODE:
            subprocess.call(experiment.split(" "))
    return True


def main():

    work = Queue()

    for (
        training_set,
        model,
        negative_sample_probability,
        rotation_augmentation,
        lr,
    ) in itertools.product(
        training_set_options,
        model_options,
        negative_sample_probability_options,
        rotation_augmentation_options,
        lr_options,
    ):

        output_dir = (
            f"output/training/{training_set}_{model}_{negative_sample_probability}_{lr}"
        )
        if rotation_augmentation == "--rotation_augmentation":
            output_dir += "_rotation"

        command = "python train.py --gpu GPU --num_epochs 75 --batch_size 64"
        +f" --training_set {training_set} --model {model}"
        +f" --negative_sample_probability {negative_sample_probability} --lr {lr}"
        +f" --save_most_recent --output_dir {output_dir} {rotation_augmentation}"
        command = command.strip()

        work.put(command)

    processes = []
    for gpu_idx in GPUS:
        p = Process(target=do_work, args=(work, gpu_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
