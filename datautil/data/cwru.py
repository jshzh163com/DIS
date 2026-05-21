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

        x, cy, py = loaddata_from_numpy(self.dataset, self.task, root_dir)
        self.people_group = people_group
        self.comb_position(x, cy, py)
        self.x = torch.tensor(self.x[:, np.newaxis, :]).float()

    def comb_position(self, x, cy, py):
        for i, peo in enumerate(self.people_group):
            index = np.where(py == peo)[0]
            ttx, ttcy = x[index], cy[index]
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x = np.vstack((self.x, ttx))
                self.labels = np.hstack((self.labels, ttcy))
