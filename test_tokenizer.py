from diffusion_policy.tokenizers.quantile_action_tokenizer import QuantileActionTokenizer
import hydra
import torch
from torch.utils.data import DataLoader
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

def mask_trajectory(trajectory, action_nums, dim_pos):
        mask = torch.ones(trajectory.shape, device=trajectory.device)
        

        dim2 = torch.arange(trajectory.shape[2]).unsqueeze(0).expand(trajectory.shape[0], trajectory.shape[2])
        mask = mask.transpose(1, 2)
        mask[dim2 >= dim_pos.unsqueeze(1), :] = 0
        mask = mask.transpose(1, 2)
        dim1 = torch.arange(trajectory.shape[1]).unsqueeze(0).expand(trajectory.shape[0], trajectory.shape[1])
        mask[dim1 > action_nums.unsqueeze(1), :] = 0
        mask[dim1 < action_nums.unsqueeze(1), :] = 1

        return trajectory * mask + tokenizer.EMPTY_ACTION_TOKEN * (1 - mask)

action_dim = 2
vocab_size = 100

dataset = PushTImageDataset(zarr_path='data/pusht/pusht_cchi_v7_replay.zarr',
                            horizon=10,
                            pad_after=7,
                            pad_before=1,
                            seed=42,
                            val_ratio=0.02,
                            max_train_episodes=90)

train_dataloader = DataLoader(dataset, batch_size=64, num_workers=8,
                              persistent_workers=False, pin_memory=True, shuffle=True)
normalizer = dataset.get_normalizer()
tokenizer = QuantileActionTokenizer(action_dim=action_dim, vocab_size=vocab_size)

# print(next(iter(train_dataloader))['obs']['agent_pos'].shape)

batch = next(iter(train_dataloader))
nactions = normalizer['action'].normalize(batch['action'])
tokenizer.fit(nactions)
trajectory = tokenizer.encode(nactions)

bsz = nactions.shape[0]

action_nums = torch.randint(
        0, 10, 
        (bsz,), device=trajectory.device
).long()

# Sample the dimension to predict
dim_nums = torch.randint(
        0, action_dim, 
        (bsz,), device=trajectory.device
).long()

maksed_trajectory = mask_trajectory(trajectory, action_nums, dim_nums)

print(maksed_trajectory)


# action_s_1 = torch.tensor([[0, 0], [2, 3], [10, 11], [20, 20]], dtype=torch.float32)
# action_s_2 = torch.tensor([[0, 0], [1, 2], [5, 6], [17, 13]], dtype=torch.float32)

# actions_seqs = torch.cat([action_s_1.unsqueeze(0), action_s_2.unsqueeze(0)], dim=0)

# tokenizer = QuantileActionTokenizer(action_dim=action_dim, vocab_size=vocab_size)

# tokenizer.fit(action_sequences=actions_seqs)

# print('Action Sequence shape:', actions_seqs.shape)


# print('Quantiles:', tokenizer.quantiles)
# print('Num Bins:', tokenizer.num_bins)
# print('Bin Width:', tokenizer.bin_widths)

# encoded = tokenizer.encode(action_sequence=actions_seqs)
# print('Encoded:', encoded)
# decoded = tokenizer.decode(token_sequence=encoded)
# print('Decoded:', decoded)