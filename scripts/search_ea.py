import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time
from datetime import datetime

import argparse
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os
from torch import autocast
from contextlib import contextmanager, nullcontext
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.data.build_dataloader import build_dataloader

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import logging
from torch.nn.functional import adaptive_avg_pool2d
from mmengine.runner import set_random_seed
# from pytorch_lightning import seed_everything
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
# from pytorch_fid.inception import InceptionV3
import copy

from EvolutionSearcher import EvolutionSearcher

# Open-Sora related imports
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import read_config
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

def str2bool(value):
    """Convert string to boolean."""
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise ValueError(f"Invalid value for boolean: {value}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/yfeng/ygcheng/src/Open-Sora/configs/opensora-v1-2/inference/sample_ea.py", # TODO
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="path to data",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=4,
        help="samples num",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--select_num",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--population_num",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--m_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--crossover_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--mutation_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--ref_latent",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ref_sigma",
        type=str,
        default='',
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--use_ddim_init_x",
        type=str2bool, # the parser does not automatically convert strings like 'false' or 'true' into actual boolean values (False or True).
        default=False,
    )
    opt = parser.parse_args()

    # TODO: Load and build models
    # ======================================================
    # Integrate Open-Sora Configurations
    # ======================================================
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = read_config(f"{opt.config}") # Load Open-Sora config file

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024)) # TODO: both ea and opensora have arg seed
    # seed_everything(opt.seed)

    # == init logger ==
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format the timestamp
    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(outpath, f"log.txt_{timestamp}"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("Inference configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # build model & load weights
    # ======================================================    
    # model = model.to(device)
    logging.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # TODO: Use Open-Sora rf sampler
    # == build scheduler ==
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)  # 采样器
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        # sampler = DDIMSampler(model)
        sampler = build_module(cfg.scheduler, SCHEDULERS)

    # dataloader_info = build_dataloader(config, opt) # TODO: Pass reference data to ea searcher

    batch_size = opt.n_samples

    if opt.dpm_solver:
        tmp_sampler = DPMSolverSampler(model)
        from ldm.models.diffusion.dpm_solver.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
        ns = NoiseScheduleVP('discrete', alphas_cumprod=tmp_sampler.alphas_cumprod)
        dpm_solver = DPM_Solver(None, ns, predict_x0=True, thresholding=False)
        skip_type = "time_uniform"
        t_0 = 1. / dpm_solver.noise_schedule.total_N  # 0.001
        t_T = dpm_solver.noise_schedule.T  # 1.0
        full_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=1000, device='cpu')
        init_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=opt.time_step, device='cpu')
        dpm_params = dict()
        full_timesteps = list(full_timesteps)
        dpm_params['full_timesteps'] = [full_timesteps[i].item() for i in range(len(full_timesteps))]
        init_timesteps = list(init_timesteps)
        dpm_params['init_timesteps'] = [init_timesteps[i].item() for i in range(len(init_timesteps))]
    else:
        dpm_params = None

    '''
    rf.sample input params:
    model: pre-trained t2v diffusion model 
    text_encoder: pre-trained text embedding model
    z: latent video, initialized as 'torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)'
    prompts: texts used for conditioning
    device: 'cpu' or 'gpu'
    additional_args: model args of multi-resolution info
    mask: read 'mask_strategy' from config; len(mask)=len(prompts), default is '[""] * len(prompts)'
    guidance_scale
    progress: show tqdm progress bar
    ''' 

    ## build EA
    t = time.time()
    searcher = EvolutionSearcher(opt=opt, model=model, text_encoder=text_encoder, vae=vae, time_step=opt.time_step, ref_latent=opt.ref_latent, ref_sigma=opt.ref_sigma, sampler=sampler, dataloader_info=None, batch_size=batch_size, device=device, dtype=dtype, dpm_params=dpm_params)
    logging.info("Integrated Open-Sora Successfully ......")
    # searcher.generate_cand_video()
    searcher.search()
    # logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

if __name__ == '__main__':
    main()
