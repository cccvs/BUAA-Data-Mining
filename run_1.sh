#!/bin/bash
python ./1-data_preprocessing/convert_csv.py
cd ./data
bash convert.sh
cd ..
python ./1-data_preprocessing/generate_ubodt.py
python ./1-data_preprocessing/match_traj.py
python ./1-data_preprocessing/post_processing.py
