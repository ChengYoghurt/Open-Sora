CUDA_VISIBLE_DEVICES=0 \
python scripts/search_ea.py \
--outdir 'outputs/search_step15' \
--config '/home/yfeng/ygcheng/src/Open-Sora/configs/opensora-v1-2/inference/sample_ea.py' \
--n_samples 6 \
--num_sample 1000 \
--time_step 15 \
--max_epochs 10 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 \
--seed 1024 \
--use_ddim_init_x false \
--use_time_transform true \
--ref_latent '/home/yfeng/ygcheng/src/Open-Sora/assets/ea/video_240p_f51.pt' \
--ref_sigma '/home/yfeng/ygcheng/src/AutoDiffusion/assets/coco2014_sigma.npy' \