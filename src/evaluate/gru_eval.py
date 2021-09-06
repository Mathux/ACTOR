import torch
from tqdm import tqdm

from src.utils.fixseed import fixseed

from src.evaluate.action2motion.evaluate import A2MEvaluation
# from src.evaluate.othermetrics.evaluation import OtherMetricsEvaluation

from torch.utils.data import DataLoader
from src.utils.tensors import collate

import os

from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model
from src.datasets.get_dataset import get_datasets


class NewDataloader:
    def __init__(self, mode, model, dataiterator, device):
        assert mode in ["gen", "rc", "gt"]
        self.batches = []
        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    classes = databatch["y"]
                    gendurations = databatch["lengths"]
                    batch = model.generate(classes, gendurations)
                    batch = {key: val.to(device) for key, val in batch.items()}
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    batch["x_xyz"] = model.rot2xyz(batch["x"].to(device),
                                                   batch["mask"].to(device))
                    batch["output"] = batch["x"]
                    batch["output_xyz"] = batch["x_xyz"]
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    batch["output_xyz"] = model.rot2xyz(batch["output"],
                                                        batch["mask"])
                    batch["x_xyz"] = model.rot2xyz(batch["x"],
                                                   batch["mask"])

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)


def evaluate(parameters, folder, checkpointname, epoch, niter):
    num_frames = 60

    # fix parameters for action2motion evaluation
    parameters["num_frames"] = num_frames
    if parameters["dataset"] == "ntu13":
        parameters["jointstype"] = "a2m"
        parameters["vertstrans"] = False  # No "real" translation in this dataset
    elif parameters["dataset"] == "humanact12":
        parameters["jointstype"] = "smpl"
        parameters["vertstrans"] = True
    else:
        raise NotImplementedError("Not in this file.")

    device = parameters["device"]
    dataname = parameters["dataset"]

    # dummy => update parameters info
    get_datasets(parameters)
    model = get_gen_model(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.outputxyz = True

    a2mevaluation = A2MEvaluation(dataname, device)
    a2mmetrics = {}

    # evaluation = OtherMetricsEvaluation(device)
    # joints_metrics = {}, pose_metrics = {}

    datasetGT1 = get_datasets(parameters)["train"]
    datasetGT2 = get_datasets(parameters)["train"]

    allseeds = list(range(niter))

    try:
        for index, seed in enumerate(allseeds):
            print(f"Evaluation number: {index+1}/{niter}")
            fixseed(seed)

            datasetGT1.reset_shuffle()
            datasetGT1.shuffle()

            datasetGT2.reset_shuffle()
            datasetGT2.shuffle()

            dataiterator = DataLoader(datasetGT1, batch_size=parameters["batch_size"],
                                      shuffle=False, num_workers=8, collate_fn=collate)
            dataiterator2 = DataLoader(datasetGT2, batch_size=parameters["batch_size"],
                                       shuffle=False, num_workers=8, collate_fn=collate)

            # reconstructedloader = NewDataloader("rc", model, dataiterator, device)
            motionloader = NewDataloader("gen", model, dataiterator, device)
            gt_motionloader = NewDataloader("gt", model, dataiterator, device)
            gt_motionloader2 = NewDataloader("gt", model, dataiterator2, device)

            # Action2motionEvaluation
            loaders = {"gen": motionloader,
                       # "recons": reconstructedloader,
                       "gt": gt_motionloader,
                       "gt2": gt_motionloader2}

            a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)

            # joints_metrics[seed] = evaluation.evaluate(model, num_classes,
            # loaders, xyz=True)
            # pose_metrics[seed] = evaluation.evaluate(model, num_classes,
            # loaders, xyz=False)

    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in a2mmetrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
