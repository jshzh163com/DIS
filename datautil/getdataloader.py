import numpy as np
from torch.utils.data import DataLoader
import sklearn.model_selection as ms

import datautil.data.cwru as cwru
import datautil.data.sqv as sqv
from datautil.dataset_utils import subdataset
from datautil.mydataloader import InfiniteDataLoader


task_act = {'CWRU': cwru, 'SQV': sqv}


def get_dig_dataloader(args):
    valid_rate = 0.2
    train_datasets, eval_datasets = [], []
    dataset_builder = task_act[args.task]

    domain_groups = args.act_people[args.dataset]
    args.domain_num = len(domain_groups)
    for domain_id, people_group in enumerate(domain_groups):
        dataset = dataset_builder.DataList(
            args, args.dataset, args.data_dir, people_group)
        if domain_id in args.test_envs:
            eval_datasets.append(dataset)
            continue

        labels = dataset.labels
        if args.split_style == 'strat':
            indices = np.arange(len(labels))
            splitter = ms.StratifiedShuffleSplit(
                2,
                test_size=valid_rate,
                train_size=1 - valid_rate,
                random_state=args.seed,
            )
            train_idx, valid_idx = next(splitter.split(indices, labels))
        else:
            indices = np.arange(len(labels))
            np.random.seed(args.seed)
            np.random.shuffle(indices)
            valid_size = int(len(labels) * valid_rate)
            train_idx, valid_idx = indices[:-valid_size], indices[-valid_size:]

        train_datasets.append(subdataset(args, dataset, train_idx))
        eval_datasets.append(subdataset(args, dataset, valid_idx))

    train_loaders = [
        InfiniteDataLoader(
            dataset=dataset,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for dataset in train_datasets
    ]

    eval_loaders = [
        DataLoader(
            dataset=dataset,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for dataset in train_datasets + eval_datasets
    ]

    return train_loaders, eval_loaders
