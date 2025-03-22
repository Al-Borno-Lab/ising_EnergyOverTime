#! /bin/bash

DATA_LOAD_PATH="./Data"
MODEL_SAVE_PATH="./Output/Model"
STATS_SAVE_PATH="./Output/Stats"
MICE=("cChR1" "fChR4" "DJC000" "DJC002" "fChR5" "fChR2")

source /home/austin/Research_Projects/Inverse-Ising-solver-main/Abigail/Analysis/cluster_scripts/venv/bin/activate

for mouse in $MICE
do
	for i in {1..101}
	do
		python3 main_gen_multi_model.py --DATA_LOAD_FILE_DIR "$DATA_LOAD_PATH" --MODEL_SAVE_FILE_DIR "$MODEL_SAVE_PATH" --JOB_NUM "$i" --MOUSE "$mouse" --TIME_SEG "pre"
		python3 main_gen_multi_model.py --DATA_LOAD_FILE_DIR "$DATA_LOAD_PATH" --MODEL_SAVE_FILE_DIR "$MODEL_SAVE_PATH" --JOB_NUM "$i" --MOUSE "$mouse" --TIME_SEG "mid"
		python3 main_gen_multi_model.py --DATA_LOAD_FILE_DIR "$DATA_LOAD_PATH" --MODEL_SAVE_FILE_DIR "$MODEL_SAVE_PATH" --JOB_NUM "$i" --MOUSE "$mouse" --TIME_SEG "active"
		python3 main_gen_multi_model.py --DATA_LOAD_FILE_DIR "$DATA_LOAD_PATH" --MODEL_SAVE_FILE_DIR "$MODEL_SAVE_PATH" --JOB_NUM "$i" --MOUSE "$mouse" --TIME_SEG "full"
		(i++)
	done
done

