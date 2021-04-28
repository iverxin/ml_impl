import numpy as np
import torch


def create_padding_mask(seq): # b,S
    # 为1的为mask掉的
    pad_idx = 0
    seq = torch.eq(seq, torch.tensor(pad_idx)).float()
    return seq[:, np.newaxis, np.newaxis, :] # mask [batch_size, num_head, seq_Len, embeddding_dim


def create_behind_mask(seq_len):
    # 为1的为mask掉的
    mask = torch.triu(torch.ones(seq_len, seq_len), 1)
    return mask


src = torch.tensor([[0,2,3],[1,0,3]])
tgt = torch.tensor([[0,2,3],[1,0,3]])
mask = create_padding_mask(src)
print(mask.shape, mask.dtype)
print(mask)
print("---------")
mask1 = create_padding_mask(tgt)
mask2 = create_behind_mask(3)
mask = torch.max(mask1, mask2)
print(mask.shape, mask.dtype)
print(mask)

