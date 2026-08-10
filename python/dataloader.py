from PIL import Image
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


def load_or_download_MNIST_(path=None, shuffle_=True, N=None, FORCED_SPLIT=None):
    pass;

    data = {}

    if N is not None and FORCED_SPLIT is None:
        FORCED_SPLIT=.2
    
    exists = path is not None and os.path.isdir(path)

    if exists:
        print("MNIST folder exists locally")
    else:
        print("MNIST local folder not found, using kagglehub")
        path = kagglehub.dataset_download("alexanderyyy/mnist-png")
        path += "\\mnist_png\\"
        print(f"dir = {path}")
        # print(path); exit()

    train_test = glob(path+"\\*")
    train_test.reverse()

    # print(train_test); exit()

    i = 0
    for folder in train_test:

        key = str(folder[-5:]).replace("\\", "")
        data[key] = []
        class_folders = glob(folder+"\\*")
        images = sum(
            ([[x, int(x[-5])] for x in glob(class_folder+"\\*")]
             for class_folder in class_folders), start=[])
        if shuffle_: shuffle(images)
        total = len(images) if N is None else min(N, len(images))

        for img, class_name in tqdm(images, desc=key, total=total):
            img_ = Image.open(img)
            pixel_data = [x/255 for x in list(img_.getdata())]
            data[key].append((pixel_data, class_name))
            i+=1

            if N is not None and i>=N: break

        if shuffle_:
            shuffle(data[key])


    # force train test split
    if (FORCED_SPLIT is not None) or (N is not None and N < 50_000):
        #SPLIT = .2 #len(data["test"]) / len(data["train"]);
        SPLIT = FORCED_SPLIT if FORCED_SPLIT is not None else .2
        all_data = data["train"] + data["test"]
        #if shuffle_: shuffle(data)
        
        #print(data)

        cutoff = int(len(all_data) * SPLIT)

        out = {
            "test" : all_data[:cutoff],
            "train": all_data[cutoff+1:]
        }
    else:
        out = data
    #out=data

    return out