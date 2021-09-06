import matplotlib.pyplot as plt
import torch

from src.datasets.get_dataset import get_dataset
from src.utils.anim import plot_3d_motion
import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def viz_ntu13(dataset, device):
    """ Generate & viz samples """
    print("Visualization of the ntu13")
    
    from src.models.rotation2xyz import Rotation2xyz
    rot2xyz = Rotation2xyz(device)
    
    realsamples = []
    pose18samples = []
    pose24samples = []

    translation = True
    dataset.glob = True
    dataset.translation = translation
    
    for i in range(1, 2):
        dataset.pose_rep = "xyz"
        x_xyz = dataset[i][0]
        realsamples.append(x_xyz)
        
        dataset.pose_rep = "rotvec"
        pose = dataset[i][0]
        mask = torch.ones(pose.shape[2], dtype=bool)

        # from src.models.smpl import SMPL
        # smplmodel = SMPL().eval().to(device)
        # import ipdb; ipdb.set_trace()
        pose24 = rot2xyz(pose[None], mask[None], pose_rep="rotvec", jointstype="smpl", glob=True, translation=translation)[0]
        pose18 = rot2xyz(pose[None], mask[None], pose_rep="rotvec", jointstype="a2m", glob=True, translation=translation)[0]
        
        translation = True
        dataset.glob = True
        dataset.translation = translation
        
        # poseT = dataset[i][0]
        # pose18T = rot2xyz(poseT[None], mask[None], pose_rep="rotvec", jointstype="action2motion", glob=True, translation=translation)[0]
        
        # import ipdb; ipdb.set_trace()
        pose18samples.append(pose18)
        pose24samples.append(pose24)

    params = {"pose_rep": "xyz"}
    for i in [0]:
        for x_xyz, title in zip([pose24samples[i], pose18samples[i], realsamples[i]], ["pose_to_24", "pose_to_18", "action2motion_18"]):
            save_path = title + ".gif"
            plot_3d_motion(x_xyz, x_xyz.shape[-1], save_path, params, title=title)
            print(f"saving {save_path}")
    

if __name__ == '__main__':
    # get device
    device = torch.device('cpu')

    # get data
    DATA = get_dataset(name="ntu13")
    dataset = DATA(split="train")
    
    viz_ntu13(dataset, device)
