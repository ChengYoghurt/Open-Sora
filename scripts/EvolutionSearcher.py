import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time

import argparse
import os
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
# from scipy import linalg

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
from opensora.schedulers.rf import timestep_transform

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)



class EvolutionSearcher(object):

    def __init__(self, opt, model, text_encoder, vae, time_step, ref_latent, ref_sigma, sampler, dataloader_info, batch_size, device, dtype, dpm_params=None):
        self.opt = opt
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.sampler = sampler
        self.time_step = time_step
        self.dataloader_info = dataloader_info
        self.batch_size = batch_size
        self.model_args = None
        # self.cfg = cfg
        ## EA hyperparameters
        self.max_epochs = opt.max_epochs
        self.select_num = opt.select_num
        self.population_num = opt.population_num
        self.m_prob = opt.m_prob
        self.crossover_num = opt.crossover_num
        self.mutation_num = opt.mutation_num
        self.num_samples = opt.num_sample
        self.ddim_discretize = "uniform"
        ## tracking variable 
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.use_ddim_init_x = opt.use_ddim_init_x

        # TODO: Load ref_latent
        self.ref_latent = torch.load(ref_latent)
        self.ref_sigma = None
        #self.ref_mu = np.load(ref_mu)
        # self.ref_sigma = np.load(ref_sigma)

        self.dpm_params = dpm_params
        self.device = device
        self.dtype = dtype

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logging.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]
    
    def is_legal_before_search(self, cand):
        cand = eval(cand)
        cand = sorted(cand, reverse=True)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['mse'] = self.get_cand_mse(cand=eval(cand))
        logging.info('cand: {}, mse: {}'.format(cand, info['mse']))

        info['visited'] = True
        return True
    
    def is_legal(self, cand):
        cand = eval(cand)
        cand = sorted(cand, reverse=True)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['mse'] = self.get_cand_mse(cand=eval(cand))
        logging.info('cand: {}, mse: {}'.format(cand, info['mse']))

        info['visited'] = True
        return True
    
    def get_random_before_search(self, num):
        logging.info('random select ........')
        self.model_args = self.init_model_args(device=self.device)
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_random(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])

            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res
    
    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            # for i in range(self.sampler.ddpm_num_timesteps):
            for i in self.sampler.get_full_timesteps(additional_args=self.model_args): # TODO
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def get_mutation_dpm(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def mutate_init_x(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            # for i in range(self.sampler.ddpm_num_timesteps):
            for i in self.sampler.get_full_timesteps(additional_args=self.model_args):
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand, reverse=True)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def sample_active_subnet(self):
        # TODO: Swap the init timesteps with rf timesteps
        # original_num_steps = self.sampler.ddpm_num_timesteps
        # use_timestep = [i for i in range(original_num_steps)]
        original_timestep = self.sampler.get_full_timesteps(additional_args=self.model_args)
        random.shuffle(original_timestep)
        use_timestep = original_timestep[:self.time_step] # time_step is set by ea searcher
        return use_timestep
    
    def sample_active_subnet_dpm(self):
        use_timestep = copy.deepcopy(self.dpm_params['full_timesteps'])
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step + 1]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))] 
        return use_timestep
    
    def get_cand_mse(self, cand=None, device='cuda'):
        cand_latent = self.generate_cand_video(cand=cand)
        # MSE Calculation
        mse_loss = F.mse_loss(cand_latent, self.ref_latent)
        print("MSE Loss:", mse_loss.item())
        return mse_loss.item()

    def init_model_args(self, device='cuda'):
        cfg = read_config(f"{self.opt.config}")  # Load Open-Sora config file

        # == load prompts ==
        prompts = cfg.get("prompt", None)
        start_idx = cfg.get("start_index", 0)
        if prompts is None:
            if cfg.get("prompt_path", None) is not None:
                prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
            else:
                prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

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

        # == multi-resolution info ==
        return prepare_multi_resolution_info(
                cfg.get("multi_resolution", None), len(prompts), image_size, num_frames, cfg.fps, device, self.dtype
            )
    
    def generate_cand_video(self, cand=None, device='cuda'):
        # exit(0)
        cfg = read_config(f"{self.opt.config}") # Load Open-Sora config file
        logger = create_logger()
        # == load prompts ==
        prompts = cfg.get("prompt", None)
        start_idx = cfg.get("start_index", 0)
        if prompts is None:
            if cfg.get("prompt_path", None) is not None:
                prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
            else:
                prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

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
        # TODO: save the computation of commonly used params
        input_size = (num_frames, *image_size)
        latent_size = self.vae.get_latent_size(input_size)

        # == prepare reference ==
        reference_path = cfg.get("reference_path", [""] * len(prompts))
        mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        # == prepare arguments ==
        fps = cfg.fps
        save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
        multi_resolution = cfg.get("multi_resolution", None)
        batch_size = cfg.get("batch_size", 1)
        num_sample = cfg.get("num_sample", 1) # Number of samples to generate per prompt. #TODO: ea opt also has arg called num_samples
        loop = cfg.get("loop", 1)
        condition_frame_length = cfg.get("condition_frame_length", 5) # Number of frames used as a conditioning input in each loop iteration.
        condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
        align = cfg.get("align", None)

        save_dir = cfg.save_dir # save path for generated video
        os.makedirs(save_dir, exist_ok=True)
        sample_name = cfg.get("sample_name", None)
        prompt_as_path = cfg.get("prompt_as_path", False)

        verbose = cfg.get("verbose", 1)
        progress_wrap = tqdm if verbose == 1 else (lambda x: x)

        # == Iter over all samples ==
        for i in progress_wrap(range(0, len(prompts), batch_size)):
            # == prepare batch prompts ==
            batch_prompts = prompts[i : i + batch_size]
            ms = mask_strategy[i : i + batch_size]
            refs = reference_path[i : i + batch_size]

            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==
            refs = collect_references_batch(refs, self.vae, image_size)

            # print(f"len(batch_prompts)={len(batch_prompts)}") # 1 batch_prompts=['a beautiful waterfall']
            # print(f"len(prompts)={len(prompts)}") # 1
            # print("num_sample=", num_sample)

            # == Iter over number of sampling for one prompt ==
            for k in range(num_sample):
                # == prepare save paths ==
                save_paths = [
                    get_save_path_name(
                        save_dir,
                        sample_name=sample_name,
                        sample_idx=start_idx + idx,
                        prompt=original_batch_prompts[idx],
                        prompt_as_path=prompt_as_path,
                        num_sample=num_sample,
                        k=k,
                    )
                    for idx in range(len(batch_prompts))
                ]
                # print(f"sample_{k},save_paths={save_paths}")

                # NOTE: Skip if the sample already exists
                # This is useful for resuming sampling VBench
                if prompt_as_path and all_exists(save_paths):
                    continue

                # == process prompts step by step ==
                # 0. split prompt
                # each element in the list is [prompt_segment_list, loop_idx_list]
                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = split_prompt(prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)

                # 1. refine prompt by openai # pass
                # 2. append score
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = append_score_to_prompts(
                        prompt_segment_list,
                        aes=cfg.get("aes", None),
                        flow=cfg.get("flow", None),
                        camera_motion=cfg.get("camera_motion", None),
                    )

                # 3. clean prompt with T5
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                # 4. merge to obtain the final prompt
                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                # == Iter over loop generation ==
                video_clips = []
                for loop_i in range(loop):
                    # == get prompt for loop i ==
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                    # == add condition frames for loop ==
                    if loop_i > 0:
                        refs, ms = append_generated(
                            vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                        )

                    # == sampling ==
                    torch.manual_seed(1024)
                    z = torch.randn(len(batch_prompts), self.vae.out_channels, *latent_size, device=device, dtype=self.dtype)
                    masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                    samples = self.sampler.sample(
                        self.model,
                        self.text_encoder,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=device,
                        additional_args=self.model_args,
                        progress=verbose >= 2,
                        mask=masks,
                        ea_timesteps=cand,
                    )
                    samples = self.vae.decode(samples.to(self.dtype), num_frames=num_frames) # TODO: return video or latent
        return samples # TODO: For now, we assume num_sample=1 and samples only content one latent video
        #             video_clips.append(samples)

        #         # == save samples ==
        #         if is_main_process():
        #             for idx, batch_prompt in enumerate(batch_prompts):
        #                 if verbose >= 2:
        #                     logger.info("Prompt: %s", batch_prompt)
        #                 save_path = save_paths[idx]
        #                 video = [video_clips[i][idx] for i in range(loop)]
        #                 for i in range(1, loop):
        #                     video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
        #                 video = torch.cat(video, dim=1)
        #                 save_path = save_sample(
        #                     video,
        #                     fps=save_fps,
        #                     save_path=save_path,
        #                     verbose=verbose >= 2,
        #                 )
        #                 if save_path.endswith(".mp4") and cfg.get("watermark", False):
        #                     time.sleep(1)  # prevent loading previous generated video
        #                     add_watermark(save_path)
        #     start_idx += len(batch_prompts)
        # logger.info("Inference finished.")
        # logger.info("Saved %s samples to %s", start_idx, save_dir)


    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        if self.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)

        else:
            if self.opt.dpm_solver:
                init_x = self.dpm_params['init_timesteps']
            else:
                init_x = make_ddim_timesteps(ddim_discr_method=self.ddim_discretize, num_ddim_timesteps=self.time_step,
                                                        num_ddpm_timesteps=self.sampler.num_timesteps, verbose=False) # TODO
            init_x = sorted(list(init_x), reverse=True)
            self.is_legal_before_search(str(init_x))
            self.candidates.append(str(init_x))
            self.get_random_before_search(self.population_num // 2)
            if self.opt.dpm_solver:
                res = self.mutate_init_x_dpm(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            else:
                res = self.mutate_init_x(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            self.candidates += res
        # # Generate videos for each candidate
        # for idx, candidate in enumerate(self.candidates):
        #     logging.info(f"Generating video for candidate {idx + 1}/{len(self.candidates)}: {candidate}")
        #     self.generate_cand_video(cand=candidate, device=self.device)
        # exit(0)
        # TODO: Update the metric evaluation method
        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['mse'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['mse'])

            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logging.info('No.{} {} mse = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['mse']))
            
            if self.epoch + 1 == self.max_epochs:
                break
            # sys.exit()
            if self.opt.dpm_solver:
                mutation = self.get_mutation_dpm(
                    self.select_num, self.mutation_num, self.m_prob)
            else:
                mutation = self.get_mutation(
                    self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num) #变异+杂交凑不足population size的部分重新随机采样

            self.epoch += 1
