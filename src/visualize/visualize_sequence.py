import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from src.datasets.get_dataset import get_dataset
from src.models.get_model import get_model
from src.utils import optutils

from src.utils.anim import plot_3d_motion_on_oneframe
from src.utils.visualize import process_to_visualize

import src.utils.fixseed  # noqa


plt.switch_backend('agg')


if __name__ == '__main__':
    # parse options
    opt, folder, checkpointname, epoch = optutils.visualize_parser()

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get data
    DATA = get_dataset(name=opt.dataset)
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

    save_path = os.path.join(folder, f"fig_{epoch}")

    action_number = 0
    actioname = dataset.action_to_action_name(action_number)
    label = dataset.action_to_label(action_number)
    print(f"Generate {actioname}..")
    
    y = torch.from_numpy(np.array([label], dtype=int)).to(device)
    motion = model.generate(y, fact=1)
    motion = process_to_visualize(motion.data.cpu().numpy(), opt.visualize)[0]
    
    print("Plot motion..")
    plot_3d_motion_on_oneframe(motion, "motion.png", opt.visualize, title=actioname)
