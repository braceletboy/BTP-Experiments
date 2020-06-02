#!/bin/bash
python run.py --download_data --use_cuda --pin_memory --num_epochs 45 --train_batch_size 32 --test_batch_size 64 --workers 4 --step_size 1