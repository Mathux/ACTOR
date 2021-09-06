import os
import imageio

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .anim import plot_3d_motion_dico, load_anim


def stack_images(real, real_gens, gen):
    nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    allframes = np.concatenate((real[:, None, ...], *[x[:, None, ...] for x in real_gens], gen), 1)
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w//30, h*nats, pix), dtype=allframes.dtype)
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate((*columns[0:nleft_cols], blackborder, *columns[nleft_cols:]), 0).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)


def generate_by_video(visualization, reconstructions, generation,
                      label_to_action_name, params, nats, nspa, tmp_path):
    # shape : (17, 3, 4, 480, 640, 3)
    # (nframes, row, column, h, w, 3)
    fps = params["fps"]

    params = params.copy()

    if "output_xyz" in visualization:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, "lengths", "y"]

    visu = {key: visualization[key].data.cpu().numpy() for key in keep}
    recons = {mode: {key: reconstruction[key].data.cpu().numpy() for key in keep}
              for mode, reconstruction in reconstructions.items()}
    gener = {key: generation[key].data.cpu().numpy() for key in keep}

    lenmax = max(gener["lengths"].max(),
                 visu["lengths"].max())

    timesize = lenmax + 5
    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format, isij):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
                pbar.update()
        if isij:
            array = np.stack([[load_anim(save_path_format.format(i, j), timesize)
                               for j in range(nats)]
                              for i in tqdm(range(nspa), desc=desc.format("Load"))])
            return array.transpose(2, 0, 1, 3, 4, 5)
        else:
            array = np.stack([load_anim(save_path_format.format(i), timesize)
                              for i in tqdm(range(nats), desc=desc.format("Load"))])
            return array.transpose(1, 0, 2, 3, 4)

    with multiprocessing.Pool() as pool:
        # Generated samples
        save_path_format = os.path.join(tmp_path, "gen_{}_{}.gif")
        iterator = ((gener[outputkey][i, j],
                     gener["lengths"][i, j],
                     save_path_format.format(i, j),
                     params, {"title": f"gen: {label_to_action_name(gener['y'][i, j])}", "interval": 1000/fps})
                    for j in range(nats) for i in range(nspa))
        gener["frames"] = pool_job_with_desc(pool, iterator,
                                             "{} the generated samples",
                                             nats*nspa,
                                             save_path_format,
                                             True)
        # Real samples
        save_path_format = os.path.join(tmp_path, "real_{}.gif")
        iterator = ((visu[outputkey][i],
                     visu["lengths"][i],
                     save_path_format.format(i),
                     params, {"title": f"real: {label_to_action_name(visu['y'][i])}", "interval": 1000/fps})
                    for i in range(nats))
        visu["frames"] = pool_job_with_desc(pool, iterator,
                                            "{} the real samples",
                                            nats,
                                            save_path_format,
                                            False)
        for mode, recon in recons.items():
            # Reconstructed samples
            save_path_format = os.path.join(tmp_path, f"reconstructed_{mode}_" + "{}.gif")
            iterator = ((recon[outputkey][i],
                         recon["lengths"][i],
                         save_path_format.format(i),
                         params, {"title": f"recons: {label_to_action_name(recon['y'][i])}",
                                  "interval": 1000/fps})
                        for i in range(nats))
            recon["frames"] = pool_job_with_desc(pool, iterator,
                                                 "{} the reconstructed samples",
                                                 nats,
                                                 save_path_format,
                                                 False)

    frames = stack_images(visu["frames"], [recon["frames"] for recon in recons.values()], gener["frames"])
    return frames


def viz_epoch(model, dataset, epoch, params, folder, writer=None):
    """ Generate & viz samples """

    # visualize with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")

    noise_same_action = params["noise_same_action"]
    noise_diff_action = params["noise_diff_action"]
    duration_mode = params["duration_mode"]
    reconstruction_mode = params["reconstruction_mode"]
    decoder_test = params["decoder_test"]

    fact = params["fact_latent"]
    figname = params["figname"].format(epoch)

    nspa = params["num_samples_per_action"]
    nats = params["num_actions_to_sample"]

    num_classes = params["num_classes"]

    # define some classes
    classes = torch.randperm(num_classes)[:nats]

    meandurations = torch.from_numpy(np.array([round(dataset.get_mean_length_label(cl.item()))
                                               for cl in classes]))

    if duration_mode == "interpolate" or decoder_test == "diffduration":
        points, step = np.linspace(-nspa, nspa, nspa, retstep=True)
        points = np.round(10*points/step).astype(int)
        gendurations = meandurations.repeat((nspa, 1)) + points[:, None]
    else:
        gendurations = meandurations.repeat((nspa, 1))

    # extract the real samples
    real_samples, mask_real, real_lengths = dataset.get_label_sample_batch(classes.numpy())
    # to visualize directly

    # Visualizaion of real samples
    visualization = {"x": real_samples.to(model.device),
                     "y": classes.to(model.device),
                     "mask": mask_real.to(model.device),
                     "lengths": real_lengths.to(model.device),
                     "output": real_samples.to(model.device)}

    # Visualizaion of real samples
    if reconstruction_mode == "both":
        reconstructions = {"tf": {"x": real_samples.to(model.device),
                                  "y": classes.to(model.device),
                                  "lengths": real_lengths.to(model.device),
                                  "mask": mask_real.to(model.device),
                                  "teacher_force": True},
                           "ntf": {"x": real_samples.to(model.device),
                                   "y": classes.to(model.device),
                                   "lengths": real_lengths.to(model.device),
                                   "mask": mask_real.to(model.device)}}
    else:
        reconstructions = {reconstruction_mode: {"x": real_samples.to(model.device),
                                                 "y": classes.to(model.device),
                                                 "lengths": real_lengths.to(model.device),
                                                 "mask": mask_real.to(model.device),
                                                 "teacher_force": reconstruction_mode == "tf"}}
    print("Computing the samples poses..")

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        # Reconstruction of the real data
        for mode in reconstructions:
            model(reconstructions[mode])  # update reconstruction dicts
        reconstruction = reconstructions[list(reconstructions.keys())[0]]

        if decoder_test == "new":
            # Generate the new data
            generation = model.generate(classes, gendurations, nspa=nspa,
                                        noise_same_action=noise_same_action,
                                        noise_diff_action=noise_diff_action,
                                        fact=fact)
        elif decoder_test == "diffaction":
            assert nats == nspa
            # keep the same noise for each "sample"
            z = reconstruction["z"].repeat((nspa, 1))
            mask = reconstruction["mask"].repeat((nspa, 1))
            lengths = reconstruction["lengths"].repeat(nspa)
            # but use other labels
            y = classes.repeat_interleave(nspa).to(model.device)
            generation = {"z": z, "y": y, "mask": mask, "lengths": lengths}
            model.decoder(generation)

        elif decoder_test == "diffduration":
            z = reconstruction["z"].repeat((nspa, 1))
            lengths = gendurations.reshape(-1).to(model.device)
            mask = model.lengths_to_mask(lengths)
            y = classes.repeat(nats).to(model.device)
            generation = {"z": z, "y": y, "mask": mask, "lengths": lengths}
            model.decoder(generation)

        elif decoder_test == "interpolate_action":
            assert nats == nspa
            # same noise for each sample
            z_diff_action = torch.randn(1, model.latent_dim, device=model.device).repeat(nats, 1)
            z = z_diff_action.repeat((nspa, 1))

            # but use combination of labels and labels below
            y = F.one_hot(classes.to(model.device), model.num_classes).to(model.device)
            y_below = F.one_hot(torch.cat((classes[1:], classes[0:1])), model.num_classes).to(model.device)
            convex_factors = torch.linspace(0, 1, nspa, device=model.device)
            y_mixed = torch.einsum("nk,m->mnk", y, 1-convex_factors) + torch.einsum("nk,m->mnk", y_below, convex_factors)
            y_mixed = y_mixed.reshape(nspa*nats, y_mixed.shape[-1])

            durations = gendurations[0].to(model.device)
            durations_below = torch.cat((durations[1:], durations[0:1]))

            gendurations = torch.einsum("l,k->kl", durations, 1-convex_factors) + torch.einsum("l,k->kl", durations_below, convex_factors)
            gendurations = gendurations.to(dtype=durations.dtype)

            lengths = gendurations.to(model.device).reshape(z.shape[0])
            mask = model.lengths_to_mask(lengths)

            generation = {"z": z, "y": y_mixed, "mask": mask, "lengths": lengths}
            model.decoder(generation)

        # Get xyz for the real ones
        visualization["output_xyz"] = model.rot2xyz(visualization["output"], visualization["mask"])

    for key, val in generation.items():
        if len(generation[key].shape) == 1:
            generation[key] = val.reshape(nspa, nats)
        else:
            generation[key] = val.reshape(nspa, nats, *val.shape[1:])

    finalpath = os.path.join(folder, figname + ".gif")
    tmp_path = os.path.join(folder, f"subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video(visualization, reconstructions, generation,
                               dataset.label_to_action_name, params, nats, nspa, tmp_path)

    print(f"Writing video {finalpath}..")
    imageio.mimsave(finalpath, frames, fps=params["fps"])

    if writer is not None:
        writer.add_video(f"Video/Epoch {epoch}", frames.transpose(0, 3, 1, 2)[None], epoch, fps=params["fps"])


def viz_dataset(dataset, params, folder):
    """ Generate & viz samples """
    print("Visualization of the dataset")

    nspa = params["num_samples_per_action"]
    nats = params["num_actions_to_sample"]

    num_classes = params["num_classes"]

    figname = "{}_{}_numframes_{}_sampling_{}_step_{}".format(params["dataset"],
                                                              params["pose_rep"],
                                                              params["num_frames"],
                                                              params["sampling"],
                                                              params["sampling_step"])

    # define some classes
    classes = torch.randperm(num_classes)[:nats]

    allclasses = classes.repeat(nspa, 1).reshape(nspa*nats)
    # extract the real samples
    real_samples, mask_real, real_lengths = dataset.get_label_sample_batch(allclasses.numpy())
    # to visualize directly

    # Visualizaion of real samples
    visualization = {"x": real_samples,
                     "y": allclasses,
                     "mask": mask_real,
                     "lengths": real_lengths,
                     "output": real_samples}

    from src.models.rotation2xyz import Rotation2xyz

    device = params["device"]
    rot2xyz = Rotation2xyz(device=device)

    rot2xyz_params = {"pose_rep": params["pose_rep"],
                      "glob_rot": params["glob_rot"],
                      "glob": params["glob"],
                      "jointstype": params["jointstype"],
                      "translation": params["translation"]}

    output = visualization["output"]
    visualization["output_xyz"] = rot2xyz(output.to(device),
                                          visualization["mask"].to(device), **rot2xyz_params)

    for key, val in visualization.items():
        if len(visualization[key].shape) == 1:
            visualization[key] = val.reshape(nspa, nats)
        else:
            visualization[key] = val.reshape(nspa, nats, *val.shape[1:])

    finalpath = os.path.join(folder, figname + ".gif")
    tmp_path = os.path.join(folder, f"subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video_sequences(visualization, dataset.label_to_action_name, params, nats, nspa, tmp_path)

    print(f"Writing video {finalpath}..")
    imageio.mimsave(finalpath, frames, fps=params["fps"])


def generate_by_video_sequences(visualization, label_to_action_name, params, nats, nspa, tmp_path):
    # shape : (17, 3, 4, 480, 640, 3)
    # (nframes, row, column, h, w, 3)
    fps = params["fps"]

    if "output_xyz" in visualization:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, "lengths", "y"]
    visu = {key: visualization[key].data.cpu().numpy() for key in keep}
    lenmax = visu["lengths"].max()

    timesize = lenmax + 5
    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
                pbar.update()
        array = np.stack([[load_anim(save_path_format.format(i, j), timesize)
                           for j in range(nats)]
                          for i in tqdm(range(nspa), desc=desc.format("Load"))])
        return array.transpose(2, 0, 1, 3, 4, 5)

    with multiprocessing.Pool() as pool:
        # Real samples
        save_path_format = os.path.join(tmp_path, "real_{}_{}.gif")
        iterator = ((visu[outputkey][i, j],
                     visu["lengths"][i, j],
                     save_path_format.format(i, j),
                     params, {"title": f"real: {label_to_action_name(visu['y'][i, j])}", "interval": 1000/fps})
                    for j in range(nats) for i in range(nspa))
        visu["frames"] = pool_job_with_desc(pool, iterator,
                                            "{} the real samples",
                                            nats,
                                            save_path_format)
    frames = stack_images_sequence(visu["frames"])
    return frames


def stack_images_sequence(visu):
    print("Stacking frames..")
    allframes = visu
    nframes, nspa, nats, h, w, pix = allframes.shape
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4, 0)).transpose(3, 1, 0, 2)
        frame = np.concatenate(columns).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)
