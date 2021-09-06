import numpy as np
import pickle as pkl
import tarfile
import os
import scipy.io as sio
from tqdm import tqdm
import src.utils.rotation_conversions as geometry
import torch

W = 960
H = 540


def get_kinect_motion(tar, videos, index):
    # skeleton loading
    video = videos[index]
    skeleton_name = video.replace("color.avi", "skeleton.mat")
    skeleton_path = os.path.join("mat_from_skeleton", skeleton_name)
    ffile = tar.extractfile(skeleton_path)
    skeleton = sio.loadmat(ffile, variable_names=["v"])["v"]
    skeleton = skeleton.reshape(-1, 25, 3)
    return skeleton


def motionto2d(motion, W=960, H=540):
    K = np.array(((540, 0, W / 2),
                  (0, 540, H / 2),
                  (0, 0, 1)))
    motion[..., 1] = -motion[..., 1]
    motion2d = np.einsum("tjk,lk->tjl", motion, K)
    nonzeroix = np.where(motion2d[..., 2] != 0)
    motion2d[nonzeroix] = motion2d[nonzeroix] / motion2d[(*nonzeroix, 2)][..., None]
    return motion2d[..., :2]


def motionto2dvibe(motion, cam):
    sx, sy, tx, ty = cam
    return (motion[..., :2] + [tx, ty]) * [W/2*sx, H/2*sy] + [W/2, H/2]


def get_kcenter(tar, videos, index):
    kmotion2d = motionto2d(get_kinect_motion(tar, videos, index))
    kboxes = np.hstack((kmotion2d.min(1), kmotion2d.max(1)))
    x1, y1, x2, y2 = kboxes.T
    kcenter = np.stack(((x1 + x2)/2, (y1 + y2)/2)).T
    return kcenter


def get_concat_goodtracks(allvibe, tar, videos, index):
    idxall = allvibe[index]
    kcenter = get_kcenter(tar, videos, index)
    tracks = np.array(list(idxall.keys()))

    if len(tracks) == 1:
        return idxall[tracks[0]], tracks

    remainingmask = np.ones(len(tracks), dtype=bool)

    currenttrack = None
    vibetracks = []
    while remainingmask.any():
        # find new track
        # first look at the closest new track in time
        candidate = np.argmin([idxall[track]["frame_ids"][0] for track in tracks[remainingmask]])
        candidate_max = idxall[tracks[remainingmask][candidate]]["frame_ids"][-1]

        # look for other candidate which intersect with the candidate (conflict)
        candidates = np.where(np.array([idxall[track]["frame_ids"][0] <= candidate_max for track in tracks[remainingmask]]))[0]

        # if the candidate is alone, take it
        if len(candidates) == 1:
            idx = np.where(remainingmask)[0][candidate]
        # if there are conflit, find the closest match
        else:
            # take the closest one in distance to the last center observed
            if currenttrack is None:  # take the kinect output
                lastbox = kcenter[0]
            else:  # take the last boxe output
                lastbox = idxall[currenttrack]["bboxes"][-1, :2]
            dists = np.linalg.norm([idxall[tracks[remainingmask][candidate]]["bboxes"][0, :2] - lastbox
                                    for candidate in candidates], axis=1)
            idx = np.where(remainingmask)[0][candidates[np.argmin(dists)]]

        # compute informations
        currenttrack = tracks[idx]
        vibetracks.append(currenttrack)
        lastframe = idxall[currenttrack]["frame_ids"][-1]

        # filter overlapping frames
        remainingmask = np.array([idxall[track]["frame_ids"][0] > lastframe for track in tracks]) & remainingmask

    goodvibe = {key: [] for key in ['pred_cam', 'orig_cam', 'pose',
                                    'betas', 'joints3d', 'bboxes', 'frame_ids']}

    for key in goodvibe:
        goodvibe[key] = np.concatenate([idxall[track][key] for track in vibetracks])

    return goodvibe, vibetracks


def interpolate_track(gvibe):
    # interpolation
    starting = np.where((gvibe["frame_ids"][1:] - gvibe["frame_ids"][:-1]) != 1)[0] + 1

    lastend = 0
    saveall = {key: [] for key in gvibe.keys() if key != "joints2d"}

    for start in starting:
        begin = start - 1
        end = start
        lastgoodidx = gvibe["frame_ids"][begin]
        firstnewgoodidx = gvibe["frame_ids"][end]

        for key in saveall.keys():
            # save the segment before the cut
            saveall[key].append(gvibe[key][lastend:begin+1])

            # extract the last good info
            lastgoodinfo = gvibe[key][begin]

            # extract the first regood info
            newfirstgoodinfo = gvibe[key][end]

            if key == "pose":  # interpolate in quaternions
                q0 = geometry.axis_angle_to_quaternion(torch.from_numpy(lastgoodinfo.reshape(24, 3)))
                q1 = geometry.axis_angle_to_quaternion(torch.from_numpy(newfirstgoodinfo.reshape(24, 3)))
                q2 = geometry.axis_angle_to_quaternion(-torch.from_numpy(newfirstgoodinfo.reshape(24, 3)))
                # Help when the interpolation is between pi and -pi
                # It avoid the problem of inverting people with global rotation
                # It is not optimal but it is better than nothing
                # newfirstgoodinfo = torch.where((torch.argmin(torch.stack((torch.linalg.norm(q0-q1, axis=1),
                # torch.linalg.norm(q0-q2, axis=1))), axis=0) == 0)[:, None], q1, q2)
                first = [q1[0], q2[0]][np.argmin((torch.linalg.norm(q0[0]-q1[0]),
                                                  torch.linalg.norm(q0[0]-q2[0])))]
                newfirstgoodinfo = q1
                newfirstgoodinfo[0] = first
                lastgoodinfo = q0

            # interpolate in between
            interinfo = []
            for x in range(lastgoodidx+1, firstnewgoodidx):
                # linear coeficient
                w2 = x - lastgoodidx
                w1 = firstnewgoodidx - x
                w1, w2 = w1/(w1+w2), w2/(w1+w2)

                inter = lastgoodinfo * w1 + newfirstgoodinfo * w2
                if key == "pose":  # interpolate in quaternions
                    # normalize the quaternion
                    inter = inter/torch.linalg.norm(inter, axis=1)[:, None]
                    inter = geometry.quaternion_to_axis_angle(inter).numpy().reshape(-1)

                interinfo.append(inter)

            saveall[key].append(interinfo)
        lastend = end

    for key in saveall.keys():
        saveall[key].append(gvibe[key][lastend:])
        saveall[key] = np.concatenate(saveall[key])

    saveall["frame_ids"] = np.round(saveall["frame_ids"]).astype(int)

    # make sure the interpolation was fine => looking at a whole frame_ids

    assert (saveall["frame_ids"] == np.arange(gvibe["frame_ids"].min(), gvibe["frame_ids"].max()+1)).all()

    return saveall


if __name__ == "__main__":
    datapath = "datasets/uestc/"
    allpath = os.path.join(datapath, "vibe_cache_all_tracks.pkl")
    oldpath = os.path.join(datapath, "vibe_cache.pkl")
    videopath = os.path.join(datapath, 'info', 'names.txt')

    kinectpath = os.path.join(datapath, "mat_from_skeleton.tar")

    allvibe = pkl.load(open(allpath, "rb"))
    oldvibe = pkl.load(open(oldpath, "rb"))

    videos = open(videopath, 'r').read().splitlines()

    tar = tarfile.open(kinectpath, "r")

    newvibelst = []
    allvtracks = []
    for index in tqdm(range(len(videos))):
        gvibe, vtracks = get_concat_goodtracks(allvibe, tar, videos, index)
        allvtracks.append(vtracks)
        newvibelst.append(interpolate_track(gvibe))

    newvibe = {key: [] for key in newvibelst[0].keys()}

    for nvibe in newvibelst:
        for key in newvibe:
            newvibe[key].append(nvibe[key])

    pkl.dump(newvibe, open("newvibe.pkl", "wb"))
