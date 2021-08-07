#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

python postprocess.py --input_fn data/naip_chesapeake_bay_2017-2018.csv --input_dir output/chesapeake-bay-3-18-2021/inference/ --output_fn output/chesapeake-bay-3-18-2021/postprocessed/all.geojson --threshold 127
