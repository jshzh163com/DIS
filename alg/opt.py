import torch


def get_optimizer(model, args, isteacher=False):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def get_scheduler(optimizer, args):
    if not args.schuse:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.max_epoch * args.steps_per_epoch,
    )
