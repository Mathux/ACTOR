import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Encoder_GRU(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames,
                 num_classes, translation, pose_rep, glob, glob_rot,
                 concatenate_time=True, latent_dim=256, num_layers=4, **kargs):
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
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Layers
        self.input_feats = self.njoints*self.nfeats + self.num_classes
        if self.concatenate_time:
            self.input_feats += 1
            
        self.feats_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)

        if self.modeltype == "cvae":
            self.mu = nn.Linear(self.latent_dim, self.latent_dim)
            self.var = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.final = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, batch):
        x, y, mask, lengths = batch["x"], batch["y"], batch["mask"], batch["lengths"]
        bs = len(y)
        x = x.permute((0, 3, 1, 2))
        x = augment_x(x, y, mask, lengths, self.num_classes, self.concatenate_time)

        # Model
        x = self.feats_embedding(x)
        x = self.gru(x)[0]
        
        # Get last valid input
        x = x[tuple(torch.stack((torch.arange(bs, device=x.device), lengths-1)))]
        
        if self.modeltype == "cvae":
            return {"mu": self.mu(x), "logvar": self.var(x)}
        else:
            return {"z": self.final(x)}


class Decoder_GRU(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames,
                 num_classes, translation, pose_rep, glob, glob_rot,
                 concatenate_time=True, latent_dim=256, num_layers=4, **kargs):
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
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Layers
        self.input_feats = self.latent_dim + self.num_classes
        if self.concatenate_time:
            self.input_feats += 1
            
        self.feats_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)

        self.output_feats = self.njoints*self.nfeats
        self.final_layer = nn.Linear(self.latent_dim, self.output_feats)
        
    def forward(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
        bs, nframes = mask.shape

        z = augment_z(z, y, mask, lengths, self.num_classes, self.concatenate_time)
        # Model
        z = self.feats_embedding(z)
        z = self.gru(z)[0]
        z = self.final_layer(z)

        # Post process
        z = z.reshape(bs, nframes, self.njoints, self.nfeats)
        # 0 for padded sequences
        z[~mask] = 0
        z = z.permute(0, 2, 3, 1)

        batch["output"] = z
        return batch
