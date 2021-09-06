import torch
import numpy as np

from ..action2motion.diversity import calculate_diversity_multimodality
from .acceleration import calculate_acceletation


class OtherMetricsEvaluation:
    """ Evaluation of some metrics in output space (not feature space):
    - Acceleration metrics
    - Reconstruction loss
    - Diversity
    - Multimodality
    (Not used in the paper)
"""
    def __init__(self, device):
        self.device = device
        
    def compute_features(self, model, motionloader, xyz=True):
        feat = "output_xyz" if xyz else "output"
        activations = []
        labels = []
        for idx, batch in enumerate(motionloader):
            batch_motion = batch[feat]
            batch_label = batch["y"]
            activations.append(batch_motion)
            labels.append(batch_label)
        activations = torch.cat(activations, dim=0)
        activations = activations.reshape(activations.shape[0], -1)
        labels = torch.cat(labels, dim=0)
        return activations, labels

    def reconstructionloss(self, motionloader, xyz=True):
        infeat = "x_xyz" if xyz else "x"
        outfeat = "output_xyz" if xyz else "output"

        sum_loss = 0
        num_loss = 0
        for batch in motionloader:
            motion_in = batch[infeat].permute(0, 3, 1, 2)
            motion_out = batch[outfeat].permute(0, 3, 1, 2)
            mask = batch["mask"]

            square_diff = (motion_in[mask] - motion_out[mask])**2
            sum_loss += square_diff.sum().item()
            num_loss += np.prod(square_diff.shape)

        rcloss = sum_loss / num_loss

        return rcloss
    
    def evaluate(self, model, num_classes, loaders, xyz=True):
        # get the xyz as well
        model.outputxyz = True
        metrics = {}
        repname = "xyz" if xyz else "pose"
        
        def print_logs(metric, key):
            print(f"Computing {metric} on the {key} loader ({repname})...")
            
        for key, loader in loaders.items():
            # acceleration
            metric = "acceleration"
            print_logs(metric, key)
            mkey = f"{metric}_{key}"
            metrics[mkey] = calculate_acceletation(loader, device=self.device, xyz=xyz)
            
            # features for diversity
            print_logs("features", key)
            feats, labels = self.compute_features(model, loader, xyz=xyz)

            # diversity and multimodality
            metric = "diversity"
            print_logs(metric, key)
            ret = calculate_diversity_multimodality(feats, labels, num_classes)
            metrics[f"diversity_{key}"], metrics[f"multimodality_{key}"] = ret

        metric = "rc_recons"
        print(f"Computing reconstruction loss ({repname})..")
        rcloss = self.reconstructionloss(loaders["recons"], xyz=xyz)
        metrics[metric] = rcloss
        return metrics
