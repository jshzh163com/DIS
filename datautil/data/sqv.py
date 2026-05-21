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
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))
        self.comb_position(x, cy, py, sy)
        self.x = self.x[:, np.newaxis, :]
        self.x = torch.tensor(self.x).float()

    def comb_position(self, x, cy, py, sy):
        for i, peo in enumerate(self.people_group):
            index = np.where(py == peo)[0]
            tx, tcy, tsy = x[index], cy[index], sy[index]
            for j, sen in enumerate(self.position):
                index = np.where(tsy == sen)[0]
                if j == 0:
                    ttx, ttcy = tx[index], tcy[index]
                else:
                    ttx = np.hstack((ttx, tx[index]))
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x, self.labels = np.vstack(
                    (self.x, ttx)), np.hstack((self.labels, ttcy))
