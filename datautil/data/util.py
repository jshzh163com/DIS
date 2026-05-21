import os
import numpy as np


def loaddata_from_numpy(dataset, task, root_dir):
    if task == 'cwru':
        x_path = os.path.join(root_dir, 'CWRU_x.npy')
        y_path = os.path.join(root_dir, 'CWRU_y.npy')
    elif task == 'sqv':
        x_path = os.path.join(root_dir, 'SQV_x.npy')
        y_path = os.path.join(root_dir, 'SQV_y.npy')
    else:
        raise ValueError("Unsupported task: {}".format(task))
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            "Missing dataset files: {} and {}".format(x_path, y_path)
        )
    x = np.load(x_path)
    ty = np.load(y_path)
    if ty.ndim != 2 or ty.shape[1] < 3:
        raise ValueError(
            "Label file must have at least three columns: class, domain, position")
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy
