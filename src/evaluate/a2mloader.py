import os
import numpy as np

from glob import glob

PATH = "/home/mathis/action2motion/eval_results/vae/filterall/ntu_rgbd_vibe/vanilla_vae_lie_mse_kld01R0/keypoint/"


class A2Mloader:
    def __init__(self, denoisedornot=True):
        labels = []
        joints = []
        
        for path in glob(os.path.join(PATH, "action*.npy")):
            name = os.path.split(path)[1]
            if "denoised" in name:
                if not denoisedornot:
                    continue
            else:
                if denoisedornot:
                    continue
            els = name.split(".")[0].split("_")

            xyz = np.load(path)
            
            y = int(els[1])
            # rep = int(els[3])
            
            labels.append(y)
            joints.append(xyz)

        self.labels = np.array(labels)
        self.joints = np.array(joints)
