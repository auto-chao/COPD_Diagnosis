#!/bin/bash

cd ./data
python processing_data.py
cd ..

python ./main_ATE.py --learning_rate 1e-6 --ATE_num 1  --dropout 0.35 --epochs 100  --batch_size 32
python ./main_LINCLS.py --learning_rate 1e-6 --ATE_num 1 --epochs 100  --batch_size 32	
python ./main_LINCLS.py --mode test --ATE_num 1 --batch_size 32	