# from PIL import Image
from glob import glob
from random import shuffle
from tqdm import tqdm
import os
import kagglehub
import gzip, struct
import numpy as np




# recursive search for all archives
_collapse_path = lambda path : glob(os.path.join(path, "**", "*"), recursive=True)


# deals with .idx file specifics
def read_archive_IDX(ospath):
    with open(ospath, "rb") as f:

        # read .idx header to get dimension and shape
        dim = struct.unpack(">I", f.read(4))[0] & 0xFF
        shape = struct.unpack(">" + "I" * dim, f.read(4 * dim))

        return np.frombuffer(f.read(), np.uint8).reshape(shape)


def load_or_download_MNIST(path=None, shuffle_=True, N=None, FORCED_SPLIT=None):
    if path is None:
        path = kagglehub.dataset_download("hojjatk/mnist-dataset")


    data = []
    count = 0

    for _dir in ("train", "t10k"):

        Xs = read_archive_IDX(
            os.path.join(path, f"{_dir}-images.idx3-ubyte")
        ).reshape(-1, 784)

        Ys = read_archive_IDX(
            os.path.join(path, f"{_dir}-labels.idx1-ubyte")
        )
            

        if N is not None:
            Xs, Ys = Xs[: max(0, N - count)],\
                     Ys[: max(0, N - count)]

        data.append(list(zip((Xs / 255).tolist(), Ys.tolist())))
        count += len(Xs)


    if (FORCED_SPLIT is not None) or (N is not None):

        both = sum(data, start = [])
        
        if shuffle_: 
            shuffle(both)

        SPLIT = FORCED_SPLIT if FORCED_SPLIT is not None else .2
        cutoff = int(len(both) * SPLIT)

        return {
            "test" : both[:cutoff],
            "train" : both[cutoff:]
        }

    else:

        if shuffle_:
            for d in data:
                shuffle(d)

        return {
            "test" : data[1],
            "train" : data[0]
        }


