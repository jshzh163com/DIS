import numpy as np
import torch

from datautil.data.util import loaddata_from_numpy
from datautil.dataset_utils import mydataset


class DataList(mydataset):
    def __init__(self, args, dataset, root_dir, people_group):
        super(DataList, self).__init__(args)
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'cwru'

        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        self.people_group = people_group
        self.position = np.sort(np.unique(sy))
        self.comb_position(x, cy, py, sy)
        self.x = torch.tensor(self.x[:, np.newaxis, :]).float()

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
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))
