import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import optutils
from ..utils.visualize import viz_epoch, viz_fake, viz_real

from ..models.get_model import get_model
from ..datasets.get_dataset import get_dataset
from ..utils.trainer import train, test

import ..utils.fixseed  # noqa


plt.switch_backend('agg')


if __name__ == '__main__':
    # parse options
    opt, folder, checkpointname, epoch = optutils.parse_load_args()
    
    # get device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get data
    DATA = get_dataset(name=opt.dataname)
    dataset = DATA(split="train", **opt.data)
    test_dataset = train_dataset = dataset
    
    # update model parameters
    opt.model.update({"num_classes": dataset.num_classes, "nfeats": dataset.nfeats, "device": device})

    # update visualize params
    opt.visualize.update({"num_classes": dataset.num_classes,
                          "num_actions_to_sample": min(opt.visualize["num_actions_to_sample"],
                                                       dataset.num_classes)})
    
    # get model
    MODEL = get_model(opt.modelname)
    model = MODEL(**opt.model)
    model = model.to(device)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)

    nexemple = 20
    latents = []
    labels = []
    generats = []
    
    print("Evaluating model..")
    keep = {"x": [], "y": [], "di": []}

    num_classes = dataset.num_classes
    # num_classes = 1
    
    for label in tqdm(range(num_classes)):
        xcp, ycp, di = dataset.get_label_sample(label, n=nexemple, return_labels=True, return_index=True)
        keep["x"].append(xcp)
        keep["y"].append(ycp)
        keep["di"].append(di)
        
        x = torch.from_numpy(xcp).to(device)
        y = torch.from_numpy(ycp).to(device)
        h = model.return_latent(x, y)
        
        # mu, var = model.encoder(x, y)
        # h = mu

        hy = torch.randn(nexemple, model.latent_dim, device=device)
        
        hcp = h.data.cpu().numpy()
        hycp = hy.data.cpu().numpy()
        
        latents.append(hcp)
        generats.append(hycp)
        
        labels.append(ycp)
        
    latents = np.array(latents)
    generats = np.array(generats)
    
    nclasses, nexemple, latent_dim = latents.shape
    labels = np.array(labels)
    all_latents = np.concatenate(latents)
    all_generats = np.concatenate(generats)

    nall_latents = len(all_latents)

    # import ipdb; ipdb.set_trace()
    print("Computing tsne..")
    from sklearn.manifold import TSNE

    all_input = np.concatenate((all_latents, all_generats))
    # tsne = TSNE(n_components=2)
    # all_vizu_concat = tsne.fit_transform(all_input)
    # import ipdb; ipdb.set_trace()
    # feats = tuple(np.argsort(all_latents.var(0))[::-1][:2])
    feats = tuple(np.argsort(all_latents.min(0)-all_latents.max(0))[::-1][:2] )
    all_vizu_concat = all_input[:, feats]
    
    all_vizu_vectors = all_vizu_concat[:nall_latents]
    all_gen_vizu_vectors = all_vizu_concat[nall_latents:]

    gen_vizu_vectors = all_gen_vizu_vectors.reshape(nclasses, nexemple, 2)
    vizu_vectors = all_vizu_vectors.reshape(nclasses, nexemple, 2)
    
    print("Plotting..")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values()) + list(mcolors.CSS4_COLORS.values())    
    for label in tqdm(range(num_classes)):
        color = colors[label]
        plt.scatter(*gen_vizu_vectors[label].T, color=color, marker="X")
        
    for label in tqdm(range(num_classes)):
        color = colors[label]
        plt.scatter(*vizu_vectors[label].T, color=color)
        
    plt.savefig("tsne_all.png")
    plt.close()

    import ipdb; ipdb.set_trace()
    """
    mean = all_vizu_vectors.mean()
    farthest = np.argsort(np.linalg.norm(mean - all_vizu_vectors, axis=1))[::-1][0]
    cl_number, exnumber = np.argwhere(np.arange(all_vizu_vectors.shape[0]).reshape(nclasses, nexemple) == farthest)[0]

    outlier_vid = keep["x"][cl_number][exnumber]
    nframe = outlier_vid.shape[-1]
    
    from ..utils.video import SaveVideo
    save_path = "outlier.mp4"

    cl_name = dataset.label_to_action_name(cl_number)
    
    with SaveVideo(save_path, opt.visualize["fps"]) as outvideo:
        for frame in range(nframe):
            outvideo += repr_to_frame(outlier_vid[..., frame], f"{cl_name} outlier", {"pose_rep": "xyz"})
    
"""
