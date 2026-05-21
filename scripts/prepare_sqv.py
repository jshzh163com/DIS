import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


CLASS_NAMES = ["IF_1", "IF_2", "IF_3", "OF_1", "OF_2", "OF_3", "N"]


def load_signal(mat_path):
    mat = loadmat(mat_path)
    if "data" in mat:
        signal = mat["data"]
    else:
        candidates = [
            value for key, value in mat.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        ]
        if not candidates:
            raise ValueError("No numeric array found in {}".format(mat_path))
        signal = candidates[0]
    return np.asarray(signal).reshape(-1)


def segment_signal(signal, segment_length, overlap):
    step = segment_length - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than segment_length")
    if len(signal) < segment_length:
        raise ValueError("Signal is shorter than segment_length")

    starts = range(0, len(signal) - segment_length + 1, step)
    return np.stack([signal[start:start + segment_length] for start in starts], axis=0)


def prepare_sqv(args):
    data_root = Path(args.mat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    labels = []
    file_count = 0
    for domain in args.domains:
        for class_id, class_name in enumerate(CLASS_NAMES):
            mat_path = data_root / "{}_{}.mat".format(class_name, domain)
            signal = load_signal(mat_path)
            segments = segment_signal(
                signal, args.segment_length, args.overlap)

            if segments.shape[0] < args.num_samples * 2:
                raise ValueError(
                    "{} has {} segments, fewer than required {}".format(
                        mat_path, segments.shape[0], args.num_samples * 2
                    )
                )

            selected = np.concatenate(
                [segments[:args.num_samples], segments[-args.num_samples:]],
                axis=0,
            )
            label = np.column_stack([
                np.full(selected.shape[0], class_id),
                np.full(selected.shape[0], file_count),
            ])

            samples.append(selected)
            labels.append(label)
            file_count += 1

    x = np.concatenate(samples, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)

    np.save(output_dir / "SQV_x.npy", x)
    np.save(output_dir / "SQV_y.npy", y)
    print("Saved {}".format(output_dir / "SQV_x.npy"))
    print("Saved {}".format(output_dir / "SQV_y.npy"))
    print("x shape: {}, y shape: {}".format(x.shape, y.shape))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare SQV npy files from mat files")
    parser.add_argument("--mat_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/SQV/")
    parser.add_argument("--domains", type=str, default="123456")
    parser.add_argument("--segment_length", type=int, default=3200)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_sqv(parse_args())
