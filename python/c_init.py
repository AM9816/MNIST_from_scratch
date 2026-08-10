from threading import Thread
from glob import glob
from tqdm import tqdm
from time import sleep
import json, struct, os

import ctypes
from ctypes import c_bool, c_double, c_int, c_float, c_long, c_bool
from ctypes import pointer as ptr, POINTER as PTR

from plotter import LivePlotter

fpoint = c_float

class _FCN_C_CLASS(ctypes.Structure): pass;



def to_c_list(data, dataType):
    c_data = (dataType * len(data))(*data)
    return c_data

islist = lambda l : isinstance(l, list) or isinstance   (l, tuple)

_activ_dict = {
    
    "relu" : 1,
    "leaky_relu" : 2
}

_default_param_file_name = "latest.params"

class C_FCN_Interface:
    def __init__(self, dll_name:str):
        pass;

        self.C = ctypes.CDLL(dll_name)
        self.register_arguments()
        self.fcn = None
        self.roundN=5
        self.data_from_c = None
        self.thread_list = []
        self.param_count = None
        self.local_param_c_list = None
        self.shared_array_size = 32
        self.dropout = None
        self.activ = None
        
    def register_arguments(self):
        self.C.INIT_FCN.argtypes = [PTR(c_int), c_int, fpoint, c_int,
                                    c_bool]
        self.C.FORWARD_BATCHED.argtypes = [PTR(_FCN_C_CLASS), PTR(fpoint),
                                           c_int, c_int, PTR(fpoint)]
        self.C.REGISTER_DATASET.argtypes = [PTR(_FCN_C_CLASS), PTR(fpoint), c_int,
                                            c_int, PTR(fpoint), c_int, c_int,
                                            c_int, c_bool]
        self.C.PRINT_DATASET.argtypes = [PTR(_FCN_C_CLASS), c_bool]
        self.C.TRAIN.argtypes = [PTR(_FCN_C_CLASS), c_int, fpoint, fpoint, fpoint,
                                 PTR(fpoint), c_int]

        self.C.INIT_FCN.restype = PTR(_FCN_C_CLASS)

        # self.C.EXPORT_FCN.argtypes = [PTR(_FCN_C_CLASS), PTR(fpoint)]
        self.C.IMPORT_FCN.argtypes = [PTR(_FCN_C_CLASS), PTR(fpoint)]
        self.C.CLEANUP.argtypes    = [PTR(_FCN_C_CLASS)]

        self.C.TEST_ACCURACY.argtypes = [PTR(_FCN_C_CLASS)]
        self.C.TEST_ACCURACY.restype  = fpoint

    def create_fcn(self, layerSizes, dropout, activ="relu", mu=.1, sigma=10, bias_0=True,
                   activ_is_id=False):
        pass;

        if not activ_is_id:
            activ = _activ_dict[activ]

        self.dropout = dropout
        self.activ = activ

        c_layerSizes = to_c_list(layerSizes, c_int)
        self.layers = layerSizes

        fcn = self.C.INIT_FCN(
            c_layerSizes, len(layerSizes),
            dropout, activ, mu, sigma, bias_0)

        self.fcn : PTR(_FCN_C_CLASS) = fcn
        #print(self.fcn); quit()

        #self.C.display()


        # count parameters
        self.param_count = 0

        for i in range(len(layerSizes) - 1):
            self.param_count += layerSizes[i + 1];
            self.param_count += layerSizes[i] * layerSizes[i+1]

        # generate local array for writing parameters to a file
        self.local_param_c_list = to_c_list([0 for _ in range(self.param_count)],
                                            fpoint)

    
    def __call__(self, x):
        if not islist(x[0]):
            x = [x]
        assert all(len(x_) == self.layers[0] for x_ in x)
        return self.forward_batched(x)

    def forward_batched(self, x):
        pass
        
        bSize = len(x)
        xSize = len(x[0])

        x_flat = []
        for x_ in x:
            for item in x_:
                x_flat.append(item)


        c_x_flat = to_c_list(x_flat, fpoint)

        out_iter = range(self.layers[-1] * bSize)
        out:PTR(fpoint) = to_c_list([0 for _ in out_iter], fpoint)

        self.C.FORWARD_BATCHED(self.fcn, c_x_flat, xSize, bSize, out)

        return [float(out[i]) for i in out_iter]

    def add_dataset(self, dataset, bsize = 2, classN=10, print_after=False):
        pass;

        

        for key in ["train", "test"]:

            if not (key in dataset.keys()):
                continue

            data = dataset[key]
            #print(data)
            Xs = []; Ys = []
            xSize = len(data[0][0])

            for X, Y in tqdm(data):
                Xs += X
                Ys.append(Y)


            Xs_flat = to_c_list(Xs, fpoint)
            Ys_flat = to_c_list(Ys, fpoint)
            
            isTrain = "train" in key.lower()
            self.C.REGISTER_DATASET(self.fcn,
                Xs_flat, xSize, len(Xs_flat),
                Ys_flat, len(Ys_flat), classN, bsize,
                isTrain
            )

            #exit()
        if print_after:
            self.C.PRINT_DATASET(self.fcn, False)

    def data_to_str(self):
        if self.data_from_c is None: return ''
        else:
            N_ = self.roundN
            return \
                f'''epoch={int(self.data_from_c[2])}, loss={round(self.data_from_c[3], N_)}, val loss = {round(self.data_from_c[4], N_)}, acc = {round(self.data_from_c[5], N_)}'''

    def _train_thread(self, epochs = 10, lr=1e-3, momentum=.8,
                            weight_decay = 0):
        
        
        
        self.C.TRAIN(self.fcn, epochs, lr, momentum, weight_decay,
                     self.data_from_c, self.shared_array_size,
                     self.local_param_c_list)
            
    def save_params(self, filename = _default_param_file_name):
        assert self.local_param_c_list is not None and self.param_count is not None,\
            "FCN interface initialized but no FCN created, cannot save to file"

        _c_params_snapshop = (fpoint * self.param_count)()
        self.C.EXPORT_FCN(self.fcn, 
                          _c_params_snapshop)

        header_out = json.dumps({
            "shape":      list(self.layers),
            "dropout":    self.dropout,
            "activation": self.activ,
            "count":      self.param_count,
            "itemsize":   ctypes.sizeof(fpoint),
        }).encode("utf-8")

        with open(filename, "wb") as f:
            # f.write(b"FCN1")
            f.write(struct.pack("<I", len(header_out)))
            f.write(header_out)


            f.write(bytes(_c_params_snapshop))
            

            # force disk write
            f.flush()
            os.fsync(f.fileno())

    # def get_consistent_params(self):
    #     pass;



    def load_params(self, filename = _default_param_file_name):
        with open(filename, "rb") as f:
            hlen, = struct.unpack("<I", f.read(4))
            metadata = json.loads(f.read(hlen))
            data = f.read()

            assert metadata["itemsize"] == ctypes.sizeof(fpoint),\
                "precision mismatch"

        if self.fcn is None:
            self.create_fcn(
                metadata["shape"], 
                metadata["dropout"], 
                metadata["activation"],
                activ_is_id=True)

        ctypes.memmove(self.local_param_c_list, data, len(data))
        self.C.IMPORT_FCN(self.fcn, self.local_param_c_list)

    def train(self, epochs=10, lr=1e-3, momentum=.8,
              weight_decay = 5e-5):
              # plot=True):


        self.thread_list.append(Thread(
            target=self._train_thread, args=(
                epochs, lr, momentum, weight_decay)    
        ))

        self.data_from_c = to_c_list(
            [-1 for _ in range(self.shared_array_size)], fpoint)

        self.thread_list[-1].start()
        #iter_ = tqdm(total=self.)
        # plotter = LivePlotter(["train loss", "test loss", "acc"]) \
        #           if plot else None



        # if self.data_from_c is not None:
            
        # if plotter is not None:
        #     plotter.addEstimateData([1,1,1])
        while self.data_from_c[0] == -1:
            pass
        while self.data_from_c[0] != 1:
            pass
            with tqdm(total=1) as pbar:

                _epoch = 0

                while self.data_from_c[1] < 1:
            #        #print([x for x in self.data])
                    #pbar.clear()
                    #pbar.update(100*round(self.data[1], 3))
                    pbar.n = (round(max(0, self.data_from_c[1]), 3))
                    pbar.set_description_str(self.data_to_str())
                    pbar.refresh()
                    sleep(.05)
                    # print("a")

                    currentEpoch = self.data_from_c[2]

                    if currentEpoch > _epoch:
                        _epoch = currentEpoch

                        pbar.set_description_str("SAVING")

                        self.save_params()

                        

                        # if _epoch == 4:
                        #     print(self.test_accuracy())
                        #     exit()
                            


                pbar.n=1
                #pbar.clear()
                #print()
                sleep(.05)

                # print(self.data_to_str())

                # print("DONE EPOCH")

                # if plotter is not None:
                #     N_=self.roundN
                #     plotter.addEstimateData([
                #         round(max(0, self.data_from_c[3]), N_) * 10,
                #         round(max(0, self.data_from_c[4]), N_) * 10,
                #         round(max(0, self.data_from_c[5]), N_)
                #     ])
                #     plotter.concludeEpoch()
                #     #plotter.forceDraw(1)

    def test_accuracy(self):
        assert self.fcn is not None, "no fcn created"
        return self.C.TEST_ACCURACY(self.fcn)
                    
    def __del__(self):
        for th in self.thread_list:
            if th.is_alive():
                th.join()