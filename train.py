import argparse
import os
import sys
import time

import numpy as np
import torch

from alg import alg, modelopera
from alg.opt import get_optimizer, get_scheduler
from datautil.getdataloader import get_dig_dataloader
from utils.util import (
    Tee,
    alg_loss_dict,
    print_environ,
    save_checkpoint,
    set_random_seed,
    train_valid_target_eval_names,
)


DATASET_CONFIGS = {
    "CWRU": {
        "data_dir": "./data/CWRU/",
        "num_classes": 10,
        "input_shape": (1, 2048),
        "domain_groups": [[i * 10 + j for j in range(10)] for i in range(4)],
    },
    "SQV": {
        "data_dir": "./data/SQV/",
        "num_classes": 7,
        "input_shape": (1, 3200),
        "domain_groups": [[i * 7 + j for j in range(7)] for i in range(6)],
    },
}


def get_args():
    parser = argparse.ArgumentParser(description="DIS training")

    parser.add_argument("--dataset", type=str,
                        default="SQV", choices=["CWRU", "SQV"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="train_output")
    parser.add_argument("--test_envs", type=int, nargs="+", default=[0])
    parser.add_argument("--split_style", type=str,
                        default="strat", choices=["strat", "random"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=120)
    parser.add_argument("--steps_per_epoch", type=int, default=2)
    parser.add_argument("--checkpoint_freq", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--use_scheduler", action="store_true")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--classifier", type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument("--layer", type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument("--dis_hidden", type=int, default=256)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--lambda_d_adv", type=float, default=0.1)
    parser.add_argument("--lambda_cls_adv", type=float, default=0.1)
    parser.add_argument("--grl_lambda_dom", type=float, default=1.0)
    parser.add_argument("--grl_lambda_cls", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--gpu_id", type=str, default="0")

    args = parser.parse_args()
    args.algorithm = "DIS"
    args.task = args.dataset
    args.N_WORKERS = args.num_workers
    args.schuse = args.use_scheduler
    args.schusech = "cos"
    args.lr_decay1 = 1.0
    args.lr_decay2 = 1.0

    args = param_init(args)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, "out_DIS.txt"))
    sys.stderr = Tee(os.path.join(args.output, "err_DIS.txt"))
    print_environ()
    print("Device: {}".format(args.device))
    return args


def param_init(args):
    config = DATASET_CONFIGS[args.dataset]
    if args.data_dir is None:
        args.data_dir = config["data_dir"]
    args.act_people = {args.dataset: config["domain_groups"]}
    args.input_shape = config["input_shape"]
    args.num_classes = config["num_classes"]
    return args


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_dig_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(args.device)
    algorithm.train()

    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)

    start_time = time.time()
    print("Train teacher net")
    opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
    sch1 = get_scheduler(opt1, args)
    algorithm.teanettrain(train_loaders, args.max_epoch,
                          opt1, sch1, args.steps_per_epoch)
    print("Teacher net training time: {:.4f}".format(time.time() - start_time))

    acc_record = {}
    acc_type_list = ["train", "valid", "target"]
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0

    print("===========start training===========")
    train_start_time = time.time()
    for epoch in range(args.max_epoch):
        for _ in range(args.steps_per_epoch):
            minibatches_device = [
                data for data in next(train_minibatches_iterator)]
            step_vals = algorithm.update(minibatches_device, opt, sch, epoch)

        if (epoch in [int(args.max_epoch * 0.7), int(args.max_epoch * 0.9)]) and not args.schuse:
            print("manually decrease lr")
            for params in opt.param_groups:
                params["lr"] = params["lr"] * 0.1

        if (epoch == args.max_epoch - 1) or (epoch % args.checkpoint_freq == 0):
            print("===========epoch {}===========".format(epoch))
            loss_msg = ",".join(["{}_loss:{:.4f}".format(
                item, step_vals[item]) for item in loss_list])
            print(loss_msg)

            acc_msg = []
            for item in acc_type_list:
                acc_record[item] = np.mean(
                    np.array([modelopera.accuracy(algorithm, eval_loaders[i])
                             for i in eval_name_dict[item]])
                )
                acc_msg.append("{}_acc:{:.4f}".format(item, acc_record[item]))
            print(",".join(acc_msg))

            if acc_record["valid"] > best_valid_acc:
                best_valid_acc = acc_record["valid"]
                target_acc = acc_record["target"]

            if acc_record["valid"] >= best_valid_acc and acc_record["target"] >= target_acc:
                target_acc = acc_record["target"]
            print("total cost time: {:.4f}".format(
                time.time() - train_start_time))

    save_checkpoint("{}_{}_{}.pkl".format(
        args.dataset, args.test_envs[0], args.algorithm), algorithm, args)

    print("valid acc: {:.4f}".format(best_valid_acc))
    print("DG result: {:.4f}".format(target_acc))
    print("Last epoch test acc: {:.4f}".format(acc_record["target"]))

    with open(os.path.join(args.output, "DIS_test.txt"), "a") as f:
        f.write("{}_test{}_seed{}_{}_alpha{}_lam{}\n".format(
            args.dataset, args.test_envs[0], args.seed, args.algorithm, args.alpha, args.lam
        ))
        f.write("total cost time:{}\n".format(
            str(time.time() - train_start_time)))
        f.write("valid acc:{:.4f}\n".format(best_valid_acc))
        f.write("target acc:{:.4f}\n".format(target_acc))
        f.write("last epoch test acc: {:.4f}\n\n".format(acc_record["target"]))
