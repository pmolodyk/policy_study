from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.autoregressive_transformer import AutoregressiveTransformer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.tokenizers.quantile_action_tokenizer import QuantileActionTokenizer

START_TOKEN = -1 # TODO Setup encoding


class AutoregressiveTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps,
            actions_vocab_size,
            n_obs_steps,
            num_inference_steps=None,
            temperature = 1.0,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            action_min_q=0.01,
            action_max_q=0.99,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # configure action tokenizer
        self.tokenizer = QuantileActionTokenizer(action_dim=action_dim, vocab_size=actions_vocab_size,
                                                 min_q=action_min_q, max_q=action_max_q)
        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = action_dim * actions_vocab_size
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = AutoregressiveTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.actions_vocab_size = actions_vocab_size
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.temperature = temperature
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    # Sample next action autoregressively TODO check
    def sample_token(self, model_output):
        probabilities = F.softmax(model_output / self.temperature, dim=-1).flatten(0, 2) # (B * seq_len * dim, Vocab)
        next_tokens = torch.multinomial(probabilities, 1) # B * 1
        return next_tokens.reshape(model_output.shape[:-1])
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model

        trajectory = self.generate_start_trajectory(size=condition_data.shape, 
                                                    dtype=condition_data.dtype,
                                                    device=condition_data.device)
        device = trajectory.device
        for a in range(self.n_action_steps):
            for d in range(self.action_dim):
                model_output = model(trajectory, torch.zeros(trajectory.shape[0], device=device), cond)
                # print('OUTPUT', model_output.shape)
                # print('COND', condition_data.shape)
                # print('TRAJ', trajectory[:, a, d].shape)
                next_token = self.sample_token(model_output.reshape(condition_data.shape[0], condition_data.shape[1], self.action_dim, self.actions_vocab_size))
                # print('NEXT', next_token.shape)
                trajectory[:, a, d] = next_token[:, a, d]

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        # Decode back to action-space
        self.tokenizer.decode(naction_pred)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = self.tokenizer.encode(nactions)
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                raise NotImplementedError()
                # start = To - 1
                # end = start + self.n_action_steps
                # trajectory = nactions[:,start:end]
        else:
            raise NotImplementedError()

        bsz = trajectory.shape[0]
        
        # Predict the noise residual
        # print('TRAJECTORY', trajectory.shape)
        model_output = self.model(trajectory, torch.zeros(bsz, device=trajectory.device), cond) # TODO remove redundant t
        # print('SIZE', model_output.shape)
        pred = F.softmax(model_output, dim=-1).reshape(bsz, self.actions_vocab_size, model_output.shape[1], self.action_dim)
        # print('PRED', pred.shape)
        # Select target tokens
        target_tokens = trajectory.clone().long()
        # print('TGT', target_tokens.shape)
        loss = self.ce_loss(input=pred, target=target_tokens)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def generate_start_trajectory(self, size, dtype, device):
        data_tensor = torch.ones(size, device=device, dtype=dtype) * self.tokenizer.EMPTY_ACTION_TOKEN
        return data_tensor
    
    # Select the token to predict during training
    def mask_trajectory(self, trajectory, action_nums, dim_pos):
        device = trajectory.device
        mask = torch.ones(trajectory.shape, device=device)

        dim2 = torch.arange(trajectory.shape[2]).unsqueeze(0).expand(trajectory.shape[0], trajectory.shape[2]).to(device)
        mask = mask.transpose(1, 2)
        mask[dim2 >= dim_pos.unsqueeze(1), :] = 0
        mask = mask.transpose(1, 2)
        dim1 = torch.arange(trajectory.shape[1]).unsqueeze(0).expand(trajectory.shape[0], trajectory.shape[1]).to(device)
        mask[dim1 > action_nums.unsqueeze(1), :] = 0
        mask[dim1 < action_nums.unsqueeze(1), :] = 1

        return trajectory * mask + self.tokenizer.EMPTY_ACTION_TOKEN * (1 - mask)
    
    # Fit tokenizer to dataset
    def fit_tokenizer(self, dataset, learning_sample=-1):
        # Select lower for memory reasons
        if learning_sample == -1:
            learning_sample = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=learning_sample)
        learning_data = self.normalizer['action'].normalize(next(iter(dataloader))['action'])
        self.tokenizer.fit(learning_data)
        print('TOKENIZER fitted!', self.tokenizer.bin_widths)