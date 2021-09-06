import torch
from tqdm import tqdm


def train_or_test(model, optimizer, iterator, device, mode="train"):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    # loss of the epoch
    dict_loss = {loss: 0 for loss in model.losses}

    with grad_env():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # Put everything in device
            batch = {key: val.to(device) for key, val in batch.items()}

            if mode == "train":
                # update the gradients to zero
                optimizer.zero_grad()

            # forward pass
            batch = model(batch)
            mixed_loss, losses = model.compute_loss(batch)
            
            for key in dict_loss.keys():
                dict_loss[key] += losses[key]

            if mode == "train":
                # backward pass
                mixed_loss.backward()
                # update the weights
                optimizer.step()
    return dict_loss


def train(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="train")


def test(model, optimizer, iterator, device):
    return train_or_test(model, optimizer, iterator, device, mode="test")
