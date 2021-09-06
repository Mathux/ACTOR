import os
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.utils.get_model_and_data import get_model_and_data
from src.utils.tensors import collate

from src.evaluate.tools import save_metrics
from src.parser.checkpoint import parser

import src.utils.fixseed  # noqa


def compute_accuracy(model, datasets, parameters):
    device = parameters["device"]
    iterators = {key: DataLoader(datasets[key], batch_size=parameters["batch_size"],
                                 shuffle=False, num_workers=8, collate_fn=collate)
                 for key in datasets.keys()}

    model.eval()
    num_labels = parameters["num_classes"]

    accuracies = {}
    with torch.no_grad():
        for key, iterator in iterators.items():
            confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
            for batch in tqdm(iterator, desc=f"Computing {key} batch"):
                # Put everything in device
                batch = {key: val.to(device) for key, val in batch.items()}
                # forward pass
                batch = model(batch)
                yhat = batch["yhat"].max(dim=1).indices
                ygt = batch["y"]
                for label, pred in zip(ygt, yhat):
                    confusion[label][pred] += 1
            accuracy = (torch.trace(confusion)/torch.sum(confusion)).item()
            accuracies[key] = accuracy
    return accuracies
        

def main():
    # parse options
    parameters, folder, checkpointname, epoch = parser()
    model, datasets = get_model_and_data(parameters)
    
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)

    accuracies = compute_accuracy(model, datasets, parameters)

    metricname = "recognition_accuracies_on_samedata_{}.yaml".format(epoch)
    
    evalpath = os.path.join(folder, metricname)
    print(f"Saving score: {evalpath}")
    save_metrics(evalpath, accuracies)


if __name__ == '__main__':
    main()
