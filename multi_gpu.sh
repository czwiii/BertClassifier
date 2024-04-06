CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train_multi_gpu.py --model_config_file config/multi_gpu.config
