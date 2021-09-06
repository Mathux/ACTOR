import os
import numpy as np
import pickle as pkl
from phspdtools import CameraParams


def splitname(name):
    subject = name[1:3]
    group = name[4:6]
    time = name[7:9]
    frame1 = name[10:14]
    frame2 = name[15:19]
    action = name[20:24]
    return subject, group, time, frame1, frame2, action


def create_phpsd_name(name):
    subject, group, time, frame1, frame2, action = splitname(name)
    phpsdname = f"subject{subject}_group{int(group)}_time{int(time)}"
    return phpsdname


def get_frames(name):
    subject, group, time, frame1, frame2, action = splitname(name)
    return int(frame1), int(frame2)


def get_action(name, coarse=True):
    subject, group, time, frame1, frame2, action = splitname(name)
    if coarse:
        return action[:2]
    else:
        return action


humanact12_coarse_action_enumerator = {
    1: "warm_up",
    2: "walk",
    3: "run",
    4: "jump",
    5: "drink",
    6: "lift_dumbbell",
    7: "sit",
    8: "eat",
    9: "turn steering wheel",
    10: "phone",
    11: "boxing",
    12: "throw",
}


humanact12_coarse_action_to_label = {x: x-1 for x in range(1, 13)}


def process_datata(savepath, posesfolder="data/PHPSDposes", datapath="data/HumanAct12", campath="data/phspdCameras"):
    data_list = os.listdir(datapath)
    data_list.sort()

    camera_params = CameraParams(campath)

    vibestyle = {"poses": [], "oldposes": [], "joints3D": [], "y": []}
    for index, name in enumerate(data_list):
        foldername = create_phpsd_name(name)
        subject = foldername.split("_")[0]
        T = camera_params.get_extrinsic("c2", subject)

        frame1, frame2 = get_frames(name)
        # subjecta, groupa, timea, frame1a, frame2a, actiona = splitname(name)

        posepath = os.path.join(posesfolder, foldername, "pose.txt")
        smplposepath = os.path.join(posesfolder, foldername, "shape_smpl.txt")
        npypath = os.path.join(datapath, name)
        joints3D = np.load(npypath)

        # take this one to get same number of frames that HumanAct12 joints .npy file
        # Otherwise we have to much frames (the registration is not perfect)
        poses = []
        goodframes = []
        with open(posepath) as f:
            for line in f.readlines():
                tmp = line.split(' ')
                frame_idx = int(tmp[0])
                if frame_idx >= frame1 and frame_idx <= frame2:
                    goodframes.append(frame_idx)
                    pose = np.asarray([float(i) for i in tmp[1:]]).reshape([-1, 3])
                    poses.append(pose)
        poses = np.array(poses)

        # if joints3D.shape[0] == (frame2 - frame1 + 1):
        #     continue

        smplposes = []
        with open(smplposepath) as f:
            for line in f.readlines():
                tmp = line.split(' ')
                frame_idx = int(tmp[0])
                if frame_idx in goodframes:
                    # pose = np.asarray([float(i) for i in tmp[1:]]).reshape([-1, 3])
                    # poses.append(pose)
                    smplparam = np.asarray([float(i) for i in tmp[1:]])
                    smplpose = smplparam[13:85]
                    smplposes.append(smplpose)
        smplposes = np.array(smplposes)

        oldposes = poses.copy()
        # rotate to the good camera
        poses = T.transform(poses)
        poses = poses - poses[0][0] + joints3D[0][0]

        # and verify that the pose correspond to the humanact12 data
        if np.linalg.norm(poses - joints3D) >= 1e-10:
            print("bad")
            continue

        assert np.linalg.norm(poses - joints3D) < 1e-10

        rotation = T.getmat4()[:3, :3]

        import pytorch3d.transforms.rotation_conversions as p3d
        import torch

        # rotate the global rotation
        global_matrix = p3d.axis_angle_to_matrix(torch.from_numpy(smplposes[:, :3]))
        smplposes[:, :3] = p3d.matrix_to_axis_angle(torch.from_numpy(rotation) @ global_matrix).numpy()

        assert poses.shape[0] == joints3D.shape[0]
        assert smplposes.shape[0] == joints3D.shape[0]

        vibestyle["poses"].append(smplposes)
        vibestyle["joints3D"].append(joints3D)

        action = get_action(name, coarse=True)
        label = humanact12_coarse_action_to_label[int(action)]
        vibestyle["y"].append(label)

    pkl.dump(vibestyle, open(savepath, "wb"))


if __name__ == "__main__":
    folder = "data/HumanAct12Poses/"
    os.makedirs(folder, exist_ok=True)
    savepath = os.path.join(folder, "humanact12poses.pkl")
    process_datata(savepath)
