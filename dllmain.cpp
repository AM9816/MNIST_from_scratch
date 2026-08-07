#include "pch.h"
#include "FCN.h"
//#include <stdio.h>
#include <iostream>
#include <vector>
#include "datadefinitions.h"
#include "random.h"
#include <thread>
#include <iomanip>


BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

#define dll extern "C" __declspec(dllexport)
#define print(x) std::cout << x << std::endl;
//#define forcount(x) for (int i = 0; i < x; i++)
#define str(x) std::to_string(x)
#define rSeed 2

fpoint ReLU(fpoint inp) {
    return inp > 0
        ? inp
        : 0
        ;
}

fpoint d_ReLU(fpoint inp) {
    return inp > 0
        ? 1
        : 0
        ;
}

#define LEAKY_RELU_GRAD .01

fpoint ReLU_leaky(fpoint inp) {
    return inp > 0
        ? inp
        : LEAKY_RELU_GRAD * inp
        ;
}

fpoint d_ReLU_leaky(fpoint inp) {
    return inp > 0
        ? 1
        : LEAKY_RELU_GRAD
        ;
}

using std::thread;

template<typename T>
void printVector(vectorList<T>& l);
void printShape(Matrixd const& m);





dll FCN* INIT_FCN(int* archData, int dSize,
    fpoint dropout = 0, int activ = RELU,
    fpoint init_mu = 0, fpoint init_sigma = .1,
    bool bias_init_0 = true) {

    vectorList<int> layerSizes;
    vectorList<int> activs;
    activs.resize(dSize);
    layerSizes.resize(dSize);
    for (int i = 0; i < dSize; i++) {
        layerSizes[i] = archData[i];
        activs[i] = activ;
    }

    FCN* fcn_ptr = new FCN(
        layerSizes, dropout, activs,
        init_mu, init_sigma, bias_init_0);

    if (fcn_ptr == nullptr) {
        print("cannot create new FCN");
        return nullptr;
    }

    fcn_ptr->display();
    fcn_ptr->self = fcn_ptr;

    return fcn_ptr;
}


//dll void CLEAR_MEMORY(FCN* fcn_ptr  ) {
//    delete fcn_ptr;
//}




dll void FORWARD_UNBATCHED(FCN* fcn_ptr, fpoint* data, int dSize,
    fpoint* out_data) {
    Matrixd X;
    X.resize(dSize, 1);
    for (int i = 0; i < dSize; i++) {
        X(i, 0) = data[i];
    }


    Matrixd out = fcn_ptr->forward(X);


    for (int i = 0; i < out.size(); i++) {
        out_data[i] = (fpoint)out(i, 0);
    }

}

//dll void GET_PARAMS(FCN* fcn_ptr, fpoint* out) {
//
//}


dll void EXPORT_FCN(FCN* fcn_ptr, fpoint* out) {
    fcn_ptr->serialize(out);
}

dll void IMPORT_FCN(FCN* fcn_ptr, fpoint* in) {
    fcn_ptr->load_params(in);
}


dll void FORWARD_BATCHED(FCN* fcn_ptr, fpoint* data, int xSize, int ySize,
    fpoint* out_data) {

    Matrixd X;
    X.resize(xSize, ySize);

    int i = 0;
    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) {

            X(x, y) = data[i];
            i++;
        }
    }



    Matrixd out = fcn_ptr->forward(X);

    //return;

    i = 0;
    for (int y = 0; y < out.cols(); y++) {
        for (int x = 0; x < out.rows(); x++) {

            out_data[i] = out(x, y);
            i++;
        }
    }


}



dll void REGISTER_DATASET(FCN* fcn_ptr,
    fpoint* Xs, int x_datapoint_spacing, int xn,
    fpoint* Ys, int yn, int classN, int bsize,
    bool isTrain) {

    //print("IS TRAIN"); print(isTrain);

    vectorList<fpoint> data;
    vectorList<Matrixd> batchData;

    for (int xi = 0; xi < xn; xi++) {

        data.push_back(Xs[xi]);

        if ((xi + 1) % x_datapoint_spacing == 0) {

            // copy data to matrix
            Matrixd X;
            X.resize(x_datapoint_spacing, 1);
            for (int i = 0; i < data.size(); i++) {
                X(i, 0) = data[i];
            }

            // record and reset container
            batchData.push_back(X);
            data.clear();
        }

        if (batchData.size() >= bsize or xi == xn - 1) {
            Matrixd Xs;
            int bsize_ = batchData.size();
            Xs.resize(x_datapoint_spacing, bsize_);
            for (int i = 0; i < bsize_; i++) {
                Xs.col(i) = batchData.at(i);
            }

            DataSet<Matrixd>& dataset = isTrain ?
                fcn_ptr->trainData : fcn_ptr->testData;

            dataset.Xs.push_back(Xs);
            batchData.clear();

        }


    }

    data.clear();
    for (int yi = 0; yi < yn; yi++) {
        data.push_back(Ys[yi]);
        if (data.size() >= bsize or yi == yn - 1) {
            int bsize_ = data.size();

            Matrixd Ys;

            // one hot encode
            Ys.resize(classN, bsize_);
            Ys.fill(0);
            int i = 0;


            for (int batch = 0; batch < bsize_; batch++) {
                fpoint y = data[batch];
                Ys.col(batch).row(y).fill(1);
                //Ys(y, batch) = 1;
            }



            DataSet<Matrixd>& dataset = isTrain ?
                fcn_ptr->trainData : fcn_ptr->testData;
            dataset.Ys.push_back(Ys);

            data.clear();
        }
    }


    fcn_ptr->datapointLength = x_datapoint_spacing;

    //printShape(fcn_ptr->trainData.Xs[0]); // exit(0);
    //printShape()


}


dll void PRINT_DATASET(FCN* fcn_ptr, bool print_actual_data = false) {
    fcn_ptr->display_dataset(print_actual_data);
}


dll void TRAIN(FCN* fcn_ptr, int epochs, fpoint lr, fpoint momentum,
    fpoint weight_decay,
    fpoint* data, int dataN) {

    fcn_ptr->train(epochs, lr, momentum, weight_decay,
        data, dataN, 8);
}


dll fpoint TEST_ACCURACY(FCN* fcn_ptr) {
    auto results = fcn_ptr->test_against_unseen(-1);
    return results[0];
}


dll void CLEANUP(FCN* fcn_ptr) {
    delete fcn_ptr;
}


dll void TEST_(FCN* fcn_ptr) {

    auto x = fcn_ptr->trainData.Xs[0];
    auto y = fcn_ptr->trainData.Ys[0];

    //exit(0);

    print(x); //print(y);

    vectorList<vectorList<Matrixd>> out = fcn_ptr->backward(x, y);

    //exit(0);

    //print("weights");
    //auto& w = out[0];
    //for (auto e : w) {
    //    printShape(e);
    //}
    print("bias");
    for (auto e : out[1]) {
        printShape(e);
    }

    //print("weights");
    //auto& w1 = fcn_ptr->weights;
    //for (auto e : w1) {
    //    printShape(e);
    //}
    print("bias");
    for (auto e : fcn_ptr->bias) {
        printShape(e);
    }


    print("weights");
    for (auto e : fcn_ptr->weights) {
        printShape(e);
    }
    for (auto e : out[0]) {
        printShape(e);
    }


    exit(0);


}



dll int works_(int x) {
    dll_sanity_check(x);
    return x;
}







