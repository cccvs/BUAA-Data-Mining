#!/bin/bash
python ./1-data_preprocessing/convert_csv_split.py
cd ./data
bash convert.sh
cd ..
# python ./1-data_preprocessing/run_stmatch.py
python ./1-data_preprocessing/run_fmm.py
python ./1-data_preprocessing/post_processing.py
