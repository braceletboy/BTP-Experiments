#!/bin/bash
#python run.py --task_number 0 --networks_file BlockCCNN.yaml --model_tag BlockCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 32 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 # mnist-task 0 and mnist-task 0 augemented
#
#
#python run.py --download_data --use_cuda --pin_memory --num_epochs 45 --train_batch_size 32 --test_batch_size 64 --workers 4 --step_size 1 # mnist-task 1 and mnist-task 1 augmented
#
#
#python run.py --task_number 2 --networks_file MCCNN.yaml --model_tag MCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 32 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 --pretrained_model ./saves/mnist_task1/best_model.pt # mnist-task 2
#
#
#python run.py --task_number 2 --networks_file MCCNN.yaml --model_tag MCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 32 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 --pretrained_model ./saves/mnist_task1_augmented/best_model.pt # mnist-task 2 augemented
#
#
#python run.py --task_number 0 --dataset fashion_mnist --networks_file BlockCCNN.yaml --model_tag BlockCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 2 3 4 6 9 --step_size 1 # fashion_mnist-task 0 augmented
#
#
#python run.py --download_data --dataset fashion_mnist --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 0 1 5 7 8 --step_size 1 # fashion_mnist-task 1 augemented
#
#
#python run.py --task_number 2 --dataset fashion_mnist --networks_file MCCNN.yaml --model_tag MCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 2 3 4 6 9 --step_size 1 --pretrained_model ./saves/fashion_mnist_task1_augmented/best_model.pt # fashion_mnist-task 2 augemented
#
#
#python run.py --task_number 0 --dataset kmnist --networks_file BlockCCNN.yaml --model_tag BlockCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 # kmnist-task 0 augemented
#
#
#python run.py --download_data --dataset kmnist --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --step_size 1 # kmnist-task 1 augemented
#
#
#python run.py --task_number 2 --dataset kmnist --networks_file MCCNN.yaml --model_tag MCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 --pretrained_model ./saves/kmnist_task1_augmented/best_model.pt # kmnist-task 2 augemented
#
#
#python run.py --task_number 0 --dataset svhn --networks_file BlockCCNN.yaml --data_folder ./data/SVHN --model_tag BlockCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 # svhn-task 0 augemented
#
#
#python run.py --download_data --dataset svhn --data_folder ./data/SVHN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --step_size 1 # svhn-task 1 augemented
#
#
#python run.py --task_number 2 --dataset svhn --networks_file MCCNN.yaml --data_folder ./data/SVHN --model_tag MCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 --pretrained_model ./saves/svhn_task1_augmented/best_model.pt # svhn-task 2 augemented
#
#
#python run.py --task_number 0 --dataset svhn_extra --networks_file BlockCCNN.yaml --data_folder ./data/SVHN --model_tag BlockCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 # svhn_extra-task 0 augemented
#
#
#python run.py --download_data --dataset svhn_extra --data_folder ./data/SVHN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --step_size 1 # svhn_extra-task 1 augemented
#
#
#python run.py --task_number 2 --dataset svhn_extra --networks_file MCCNN.yaml --data_folder ./data/SVHN --model_tag MCCNN --use_cuda --pin_memory --num_epochs 45 --train_batch_size 64 --test_batch_size 64 --workers 4 --labels 5 6 7 8 9 --step_size 1 --pretrained_model ./saves/svhn_extra_task1_augmented/best_model.pt # svhn_extra-task 2 augemented