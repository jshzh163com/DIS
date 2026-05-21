class mydataset(object):
    def __init__(self, args):
        self.x = None
        self.labels = None
        self.args = args

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return len(self.x)


class subdataset(mydataset):
    def __init__(self, args, dataset, indices):
        super(subdataset, self).__init__(args)
        self.x = dataset.x[indices]
        self.labels = dataset.labels[indices]
