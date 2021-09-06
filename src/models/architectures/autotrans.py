from .transformer import Encoder_TRANSFORMER as Encoder_AUTOTRANS  # noqa

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .tools.transformer_layers import PositionalEncoding
from .tools.transformer_layers import TransformerDecoderLayer


# taken from joeynmt repo
def subsequent_mask(size: int):
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def augment_x(x, y, mask, lengths, num_classes, concatenate_time):
    bs, nframes, njoints, nfeats = x.size()
    x = x.reshape(bs, nframes, njoints*nfeats)
    if len(y.shape) == 1:  # can give on hot encoded as input
        y = F.one_hot(y, num_classes)
    y = y.to(dtype=x.dtype)
    y = y[:, None, :].repeat((1, nframes, 1))

    if concatenate_time:
        # Time embedding
        time = mask * 1/(lengths[..., None]-1)
        time = (time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :])[:, 0]
        time = time[..., None]
        x_augmented = torch.cat((x, y, time), 2)
    else:
        x_augmented = torch.cat((x, y), 2)
    return x_augmented


def augment_z(z, y, mask, lengths, num_classes, concatenate_time):
    if len(y.shape) == 1:  # can give on hot encoded as input
        y = F.one_hot(y, num_classes)
    y = y.to(dtype=z.dtype)
    # concatenete z and y and repeat the input
    z_augmented = torch.cat((z, y), 1)[:, None].repeat((1, mask.shape[1], 1))

    # Time embedding
    if concatenate_time:
        time = mask * 1/(lengths[..., None]-1)
        time = (time[:, None] * torch.arange(time.shape[1], device=z.device)[None, :])[:, 0]
        z_augmented = torch.cat((z_augmented, time[..., None]), 2)
        
    return z_augmented


class Decoder_AUTOTRANS(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 concatenate_time=True, positional_encoding=True, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4,
                 dropout=0.1, emb_dropout=0.1, teacher_forcing=True, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.concatenate_time = concatenate_time
        self.positional_encoding = positional_encoding
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.teacher_forcing = teacher_forcing
        
        self.input_feats = self.latent_dim + self.num_classes
        self.input_feats_x = self.njoints*self.nfeats + self.num_classes
        if self.concatenate_time:
            self.input_feats += 1
            self.input_feats_x += 1

        self.embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.embedding_x = nn.Linear(self.input_feats_x, self.latent_dim)
            
        self.output_feats = self.njoints*self.nfeats
        
        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([TransformerDecoderLayer(size=self.latent_dim,
                                                             ff_size=self.ff_size,
                                                             num_heads=self.num_heads,
                                                             dropout=self.dropout)
                                     for _ in range(self.num_layers)])

        self.pe = PositionalEncoding(self.latent_dim)
        self.layer_norm = nn.LayerNorm(self.latent_dim, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=self.emb_dropout)
        self.output_layer = nn.Linear(self.latent_dim, self.output_feats, bias=False)
        
    def forward(self, batch):
        z, y, mask = batch["z"], batch["y"], batch["mask"]
        lengths = mask.sum(1)
        
        lenseqmax = mask.shape[1]
        bs, njoints, nfeats = len(z), self.njoints, self.nfeats
        
        z_augmented = augment_z(z, y, mask, lengths, self.num_classes, self.concatenate_time)
        src = self.embedding(z_augmented)
        
        src_mask = mask.unsqueeze(1)
        
        # Check if using teacher forcing or not
        # if it is allowed and possible
        teacher_forcing = self.teacher_forcing and "x" in batch
        # in eval mode, by default it it not unless it is "forced"
        teacher_forcing = teacher_forcing and (self.training or batch.get("teacher_force", False))
            
        if teacher_forcing:
            x = batch["x"].permute((0, 3, 1, 2))
            # shift the input
            x = torch.cat((x.new_zeros((x.shape[0], 1, *x.shape[2:])), x[:, :-1]), axis=1)
            # Embedding of the input
            x_augmented = augment_x(x, y, mask, lengths, self.num_classes, self.concatenate_time)
            trg = self.embedding_x(x_augmented)
            trg_mask = (mask[:, None] * subsequent_mask(lenseqmax).type_as(mask))
            # shape: torch.Size([48, 183, 183])
            
            if self.positional_encoding:
                trg = self.pe(trg)
            trg = self.emb_dropout(trg)

            val = trg
            for layer in self.layers:
                val = layer(val, src, src_mask=src_mask, trg_mask=trg_mask)
                
            val = self.layer_norm(val)
            val = self.output_layer(val)

            # pad the output
            val[~mask] = 0
            
            val = val.reshape((bs, lenseqmax, njoints, nfeats))
            batch["output"] = val.permute(0, 2, 3, 1)
        else:
            # Create the first input x/src_mask
            x = torch.Tensor.new_zeros(z, (bs, 1, njoints, nfeats))
            for index in range(lenseqmax):
                # change it to speed up
                current_mask = mask[:, :index+1]
                x_augmented = augment_x(x, y, current_mask, lengths,
                                        self.num_classes, self.concatenate_time)
                trg = self.embedding_x(x_augmented)
                trg_mask = (current_mask[:, None] * subsequent_mask(index+1).type_as(mask))

                if self.positional_encoding:
                    trg = self.pe(trg)
                trg = self.emb_dropout(trg)

                val = trg
                for layer in self.layers:
                    val = layer(val, src, src_mask=src_mask, trg_mask=trg_mask)

                val = self.layer_norm(val)
                val = self.output_layer(val)

                # pad the output
                val[~current_mask] = 0
                val = val.reshape((bs, index+1, njoints, nfeats))

                # extract the last output
                last_out = val[:, -1]
                # concatenate it to input x
                x = torch.cat((x, last_out[:, None]), 1)
            # remove the dummy first input (BOS)
            batch["output"] = x[:, 1:].permute(0, 2, 3, 1)
        return batch
