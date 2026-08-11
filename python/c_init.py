from threading import Thread
from glob import glob
from tqdm import tqdm
from time import sleep
import json, struct, os

import ctypes
from ctypes import c_bool, c_double, c_int, c_float, c_long, c_bool
from ctypes import pointer as ptr, POINTER as PTR

# from plotter import LivePlotter

# must match fpoint typedef at compilation
fpoint = c_float

# required to allow python to store a pointer to arbitrary C++ object
class _FCN_C_CLASS(ctypes.Structure): pass;

def to_c_list(data, dataType):
    c_data = (dataType * len(data))(*data)
    return c_data

islist = lambda l : isinstance(l, list) or isinstance(l, tuple)

activ_dict = {    
    "relu" : 1,
    "leaky_relu" : 2
}

# default file name to be used when saving parameters
_default_param_file_name = "latest.params"

# class to bridge C++ implementation with python
# dll exposed functions are never to be called directly, only 
# using the class methods of this function, reads dll from
# file path
class C_FCN_Interface:
    def __init__(self, dll_file_path:str):
        pass;

        self.CLib = ctypes.CDLL(dll_file_path)
        self.register_arguments()

        self.fcn = None


        # data written to from C++ to share performance metrics
        # e.g. loss, accuracy, epoch number ect
        self.cData = None 
        # how many digits to round said metrics too
        self.roundN=5 
        # allows concurrent status reporting and training
        # ignores GIL as python 'sleeps' when C++ runs 
        self.threadList = []
        # required for saving and loading parameters
        self.paramCount = None
        self.sharedArraySize = 32
        self.dropout = None
        self.activ = None
        
    # required as C++ is statically typed
    def register_arguments(self):
        self.CLib.INIT_FCN.argtypes = [PTR(c_int), c_int, fpoint, c_int,
                                       c_bool]
        self.CLib.INIT_FCN.restype = PTR(_FCN_C_CLASS)
        self.CLib.FORWARD_BATCHED.argtypes = [PTR(_FCN_C_CLASS), PTR(fpoint),
                                           c_int, c_int, PTR(fpoint)]
        self.CLib.REGISTER_DATASET.argtypes = [PTR(_FCN_C_CLASS), PTR(fpoint), c_int,
                                            c_int, PTR(fpoint), c_int, c_int,
                                            c_int, c_bool]
        self.CLib.PRINT_DATASET.argtypes = [PTR(_FCN_C_CLASS), c_bool]
        self.CLib.TRAIN.argtypes = [PTR(_FCN_C_CLASS), c_int, fpoint, fpoint, fpoint,
                                 PTR(fpoint), c_int]

        

        # self.C.EXPORT_FCN.argtypes = [PTR(_FCN_C_CLASS), PTR(fpoint)]
        self.CLib.IMPORT_FCN.argtypes = [PTR(_FCN_C_CLASS), PTR(fpoint)]
        self.CLib.CLEANUP.argtypes    = [PTR(_FCN_C_CLASS)]

        self.CLib.TEST_ACCURACY.argtypes = [PTR(_FCN_C_CLASS)]
        self.CLib.TEST_ACCURACY.restype  = fpoint



    def create_fcn(self, layerSizes, dropout, activ="relu", 
                   biasInit0=True, activIsID=False):
        pass;

        if not activIsID:
            activ = activ_dict[activ]

        self.dropout = dropout
        self.activ = activ

        c_layerSizes = to_c_list(layerSizes, c_int)
        self.layers = layerSizes

        fcn = self.CLib.INIT_FCN(
            c_layerSizes, len(layerSizes),
            dropout, activ, biasInit0)

        self.fcn = fcn

        # count parameters to allow easy saving to a file
        self.paramCount = 0

        for i in range(len(layerSizes) - 1):
            self.paramCount += layerSizes[i + 1];
            self.paramCount += layerSizes[i] * layerSizes[i+1]

    # allow object to be called like a function (Pytorch style)
    def __call__(self, x):
        if not islist(x[0]):
            x = [x]
        assert all(len(x_) == self.layers[0] for x_ in x)
        return self.forward_batched(x)


    def forward_batched(self, x):
        pass
        
        bSize = len(x)
        xSize = len(x[0])

        # flatten and write to C style array
        xFlat = []
        for batch in x:
            for feature in batch:
                xFlat.append(feature)
        xFlat_c = to_c_list(xFlat, fpoint)


        outIterator = range(self.layers[-1] * bSize)
        out = to_c_list([0 for _ in outIterator], fpoint)

        self.CLib.FORWARD_BATCHED(self.fcn, xFlat_c, xSize, bSize, out)

        # turn result back to python list
        return [float(out[i]) for i in outIterator]

    # copy dataset loaded to C++
    def add_dataset(self, dataset, bsize = 2, classN=10, 
                    print_after=False):
        pass;

        

        for key in ["train", "test"]:

            if not (key in dataset.keys()):
                continue

            data = dataset[key]

            Xs = []; Ys = []
            xSize = len(data[0][0])

            for X, Y in tqdm(data):
                Xs += X
                Ys.append(Y)


            Xs_flat = to_c_list(Xs, fpoint)
            Ys_flat = to_c_list(Ys, fpoint)
            
            isTrain = "train" in key.lower()
            self.CLib.REGISTER_DATASET(self.fcn,
                Xs_flat, xSize, len(Xs_flat),
                Ys_flat, len(Ys_flat), classN, bsize,
                isTrain
            )

            #exit()
        if print_after:
            self.CLib.PRINT_DATASET(self.fcn, False)

    # read shared data and convert to human readable string
    # technically this may read some elements of cData from iteration
    # n and some from n+1 ect, however this makes little difference in practice
    def data_to_str(self):
        if self.cData is None: return ''
        else:
            N_ = self.roundN
            return \
                f'''epoch={int(self.cData[2])}, loss={round(self.cData[3], N_)}, val loss = {round(self.cData[4], N_)}, acc = {round(self.cData[5], N_)}'''

    # seperate thread to allow python to report statistics while
    # C++ carries out heavy computation
    def _train_thread(self, epochs = 10, lr=1e-3, momentum=.8,
                            weight_decay = 0):
        
        self.CLib.TRAIN(self.fcn, epochs, lr, momentum, weight_decay,
                     self.cData, self.sharedArraySize)
            
    # save parameters to given file, custom but simple byte dump 
    # with very small header detailing architecture, dropout, 
    # activation functions, parameter count and individual parameter size
    def save_params(self, filename = _default_param_file_name):

        c_params_snapshop = (fpoint * self.paramCount)()
        
        # EXPORT_FCN uses a mutex lock to prevent parameters being
        # updated while they are read here, C++ waits for python in this case
        self.CLib.EXPORT_FCN(self.fcn, 
                          c_params_snapshop)

        header_out = json.dumps({
            "shape":      list(self.layers),
            "dropout":    self.dropout,
            "activation": self.activ,
            "count":      self.paramCount,
            "itemsize":   ctypes.sizeof(fpoint),
        }).encode("utf-8")

        with open(filename, "wb") as f:
            # f.write(b"FCN1")
            f.write(struct.pack("<I", len(header_out)))
            f.write(header_out)


            f.write(bytes(c_params_snapshop))
            

            # force disk write
            f.flush()
            os.fsync(f.fileno())

    # def get_consistent_params(self):
    #     pass;


    # load parameters from given file path
    def load_params(self, filename = _default_param_file_name):
        with open(filename, "rb") as f:
            hlen, = struct.unpack("<I", f.read(4))
            metadata = json.loads(f.read(hlen))
            data = f.read()

            assert metadata["itemsize"] == ctypes.sizeof(fpoint),\
                "precision mismatch"

        # if self.fcn is None:
        self.create_fcn(
            metadata["shape"], 
            metadata["dropout"], 
            metadata["activation"],
            activIsID=True)

        local_c_array = to_c_list([0 for _ in range(self.paramCount)],
                                            fpoint)
        ctypes.memmove(local_c_array, data, len(data))

        # concurrently safe from C++ (using mutex)
        self.CLib.IMPORT_FCN(self.fcn, local_c_array)


    
    def train(self, epochs=10, lr=1e-3, momentum=.8,
              weight_decay = 5e-5):
              # plot=True):

        thread = Thread(
            target=self._train_thread, args=(
                epochs, lr, momentum, weight_decay))

        self.threadList.append(thread)

        # -1 initially to signal that C++ side has not reported anything yet
        self.cData = to_c_list(
            [-1 for _ in range(self.sharedArraySize)], fpoint)

        thread.start()

        # unoptimal busy waiting, however this happens only for 
        # a few ms while C++ initializes training parameters
        while self.cData[0] == -1:
            pass

        # while C++ has not reported that the training is finished .. 
        while self.cData[0] != 1:
            pass
            with tqdm(total=1) as pbar:

                _epoch = 0

                while self.cData[1] < 1:
                    # .. read and report current epoch progress
                    pbar.n = (round(max(0, self.cData[1]), 3))
                    # and other performance metrics
                    pbar.set_description_str(self.data_to_str())
                    pbar.refresh()

                    # avoid busy waiting 
                    sleep(.05)

                    currentEpoch = self.cData[2]

                    # save parameters per epoch, blocking C++ side
                    if currentEpoch > _epoch:
                        _epoch = currentEpoch
                        pbar.set_description_str("SAVING")
                        self.save_params()
                            


                pbar.n=1



    def test_accuracy(self):
        assert self.fcn is not None, "no fcn created"
        return float(self.CLib.TEST_ACCURACY(self.fcn))
               
    # safe exit threads
    def __del__(self):
        for th in self.threadList:
            if th.is_alive():
                th.join()