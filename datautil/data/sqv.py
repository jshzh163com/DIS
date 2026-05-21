from datautil.data.util import *
from datautil.dataset_utils import mydataset
import numpy as np
import torch


class DataList(mydataset):
    def __init__(self, args, dataset, root_dir, people_group):
        super(DataList, self).__init__(args)
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'sqv'
        x, cy, py = loaddata_from_numpy(self.dataset, self.task, root_dir)
        self.people_group = people_group
        self.comb_position(x, cy, py)
        self.x = self.x[:, np.newaxis, :]
        self.x = torch.tensor(self.x).float()

    def comb_position(self, x, cy, py):
        for i, peo in enumerate(self.people_group):
            index = np.where(py == peo)[0]
            ttx, ttcy = x[index], cy[index]
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x, self.labels = np.vstack(
                    (self.x, ttx)), np.hstack((self.labels, ttcy))
