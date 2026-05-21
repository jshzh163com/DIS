import torch


def accuracy(network, loader):
    correct = 0
    total = 0
    device = getattr(network, "args", None)
    device = getattr(device, "device",
                     "cuda" if torch.cuda.is_available() else "cpu")

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device).float()
            y = data[1].to(device).long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total
