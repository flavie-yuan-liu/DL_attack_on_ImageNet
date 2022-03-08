import torch
import torchmetrics


def model_accuracy(dataset, model, device='cpu'):
    metric = torchmetrics.Accuracy()
    metric.to(device)
    model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            model = model.to(device)
            pred = model(x)
            acc = metric(pred, y)
        acc = metric.compute()
    metric.reset()
    return acc
