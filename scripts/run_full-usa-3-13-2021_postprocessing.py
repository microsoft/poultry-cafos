"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.


Script for running postprocessing for the full-usa run by state.
"""
import os
import subprocess
from multiprocessing import Process, Queue

NUM_PROCESSES = 64
SPLITS_DIR = "data/naip_most_recent_100cm_by_state/"


def do_work(work):
    while not work.empty():
        experiment = work.get()
        print(experiment)
        subprocess.call(experiment.split(" "))
    return True


def main():

    fns = os.listdir(SPLITS_DIR)
    work = Queue()

    for fn in fns:
        output_fn = fn.replace(".csv", "")
        command = "python postprocess.py"
        +f" --input_fn data/naip_most_recent_100cm_by_state/{fn}"
        +" --output_fn output/full-usa-3-13-2021/postprocessed/"
        +f"full-usa_{output_fn}.geojson"
        +" --blob_root_dir output/full-usa-3-13-2021/inference/ --threshold 127"
        work.put(command)

    processes = []
    for i in range(NUM_PROCESSES):
        p = Process(target=do_work, args=(work,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
