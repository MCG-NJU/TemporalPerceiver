CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=10260 --use_env main.py --dataset kineticsGEBD --bc_ratio 0.8 --compress_ratio 0.6 --enc_layers 6 
