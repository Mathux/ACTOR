import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.utils.trainer import train, test
from src.utils.tensors import collate
from src.utils.get_model_and_data import get_model_and_data

from src.parser.recognition import training_parser
import src.utils.fixseed  # noqa


def do_epochs(model, datasets, parameters, optimizer, writer):
    train_dataset = datasets["train"]
    train_iterator = DataLoader(train_dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=8, collate_fn=collate)

    if "test" in datasets:
        test_dataset = datasets["test"]
        test_iterator = DataLoader(test_dataset, batch_size=parameters["batch_size"],
                                   shuffle=False, num_workers=8, collate_fn=collate)

    for epoch in range(1, parameters["num_epochs"]+1):
        train_dict_loss = train(model, optimizer, train_iterator, model.device)
        if "test" in datasets:
            test_dict_loss = test(model, optimizer, test_iterator, model.device)
        
        for key in train_dict_loss.keys():
            train_dict_loss[key] /= len(train_iterator)
            if "test" in datasets:
                test_dict_loss[key] /= len(test_iterator)
            writer.add_scalar(f"Loss/{key}", train_dict_loss[key], epoch)
            if "test" in datasets:
                writer.add_scalar(f"TestLoss/{key}", test_dict_loss[key], epoch)

        if "test" in datasets:
            print(f"Epoch {epoch}, train losses: {train_dict_loss}, test_loses: {test_dict_loss}")
        else:
            print(f"Epoch {epoch}, train losses: {train_dict_loss}")

        if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
            checkpoint_path = os.path.join(parameters["folder"],
                                           'checkpoint_{:04d}.pth.tar'.format(epoch))
            print('Saving checkpoint {}'.format(checkpoint_path))
            torch.save(model.state_dict(), checkpoint_path)

        writer.flush()


def main():    
    # parse options
    parameters = training_parser()

    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])

    model, datasets = get_model_and_data(parameters)

    datasets.pop("test")

    if parameters["dataset"] == "uestcpartial":
        dt = datasets["train"]
        normal_length = dt._oldN
        realratio = parameters["realratio"]
        withgen = parameters["withgen"]
        withreal = parameters["withreal"]
        print(f"Real ratio: {realratio}, withgen: {withgen==1}, withreal: {withreal==1}")
        expected = normal_length*realratio/100 * (2 if (withgen == 1) and (withreal == 1) else 1)
        print(f"Normal len: {len(dt)}, expected: {expected}")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    do_epochs(model, datasets, parameters, optimizer, writer)

    writer.close()


if __name__ == '__main__':
    main()
