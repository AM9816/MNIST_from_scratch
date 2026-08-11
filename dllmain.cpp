#include "pch.h"
#include "FCN.h"
#include "datadefinitions.h"
#include "random.h"


// to expose python endpoint
// all exposed functions are in UPPERCASE
#define dll extern "C" __declspec(dllexport)
    

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





nn::Activation activ_id_to_enum(int _id) {
    switch (_id) {
    case 1:
        return nn::Activation::RELU;
    case 2:
        return nn::Activation::LEAKY_RELU;
    default:
        return nn::Activation::IDENTITY;
    }
}



// initialise network using parameters from python
// returns pointer to constructed object (C style) to allow
// python to access said object
dll FCN* INIT_FCN(
    int* archData, int dSize,
    fpoint dropout = 0, int activId = 1,
    fpoint initMu = 0, fpoint initSigma = .1,
    bool biasInitially0 = true) {


    vectorList<int> layerSizes;
    vectorList<nn::Activation> activs;
    activs.resize(dSize);
    layerSizes.resize(dSize);
    
    for (int i = 0; i < dSize; i++) {
        layerSizes[i] = archData[i];
        activs[i] = activ_id_to_enum(activId);
    }

    FCN* fcnPTR = new FCN(
        layerSizes, dropout, activs,
        biasInitially0);

    if (fcnPTR == nullptr) {
        print("cannot create new FCN");
        return nullptr;
    }

    fcnPTR->display_architecture();
    //fcnPTR->self = fcnPTR;

    return fcnPTR;
}

dll void EXPORT_FCN(FCN* fcnPTR, fpoint* out) {
    fcnPTR->serialize(out);
}
dll void IMPORT_FCN(FCN* fcnPTR, fpoint* in) {
    fcnPTR->load_params(in);
}

// regular neural network evaluation without 
// gradient tracking, writes to outDataPTR
dll void FORWARD_BATCHED(
    FCN* fcnPTR, fpoint* data, 
    int xSize, int ySize,
    fpoint* outDataPTR) {


    // fast memory copy
    //Matrix X = Eigen::Map<Matrix>(data, xSize, ySize);

    // manual elemwise copy
    Matrix X;
    X.resize(xSize, ySize);
    int i = 0;
    for (int y = 0; y < ySize; y++) {
    for (int x = 0; x < xSize; x++) {
        X(x, y) = data[i]; i++; }}

    Matrix out = fcnPTR->forward(X);

    //int i = 0;
    i = 0;
    for (int y = 0; y < out.cols(); y++) {
        for (int x = 0; x < out.rows(); x++) {
            outDataPTR[i] = out(x, y);
            i++;
        }
    }


}
//Matrix X = Eigen::Map<Matrix>(&data[0], data.size(), 1);




dll void REGISTER_DATASET(FCN* fcnPTR,
    fpoint* Xs, int xSpacing, int xN,
    fpoint* Ys, int yN, int classN, int bsize,
    bool isTrain) {

    nn::DataSet<Matrix>& dataset = isTrain ?
        fcnPTR->trainData : fcnPTR->testData;

    int offset = 0;

    for (int i = 0; i < yN; i += bsize) {

        // handle the last batch being smaller than bsize
        int chunkSize = std::min(bsize, yN - i);

        Matrix X = Eigen::Map<Matrix>(
            Xs + offset,
            xSpacing, chunkSize);


        // one hot encode
        Matrix Y = Matrix::Zero(classN, chunkSize);
        for (int _i = 0; _i < chunkSize; _i++) {
            int label = (int)Ys[i + _i];
            Y(label, _i) = 1.f;
        }



        dataset.Ys.push_back(Y);
        dataset.Xs.push_back(X);
        offset += chunkSize * xSpacing;

    }

    fcnPTR->datapointLength = xSpacing;

}


// allows python end to make check if batch sizes 
// and train test splits are correct
dll void PRINT_DATASET(FCN* fcnPTR, bool print_actual_data = false) {
    fcnPTR->display_dataset(print_actual_data);
}


dll void TRAIN(
    FCN* fcnPTR, int epochs, 
    fpoint lr, fpoint momentum,
    fpoint weightDecay,
    fpoint* data, int dataN) {

    fcnPTR->train(epochs, lr, momentum, weightDecay,
        data, dataN, nn::THREADS_TO_USE, true);
}


dll fpoint TEST_ACCURACY(FCN* fcnPTR) {
    auto results = fcnPTR->test_against_unseen(-1);
    return results[0];
}


dll void CLEANUP(FCN* fcnPTR) {
    delete fcnPTR;
}


// python side sanity checks, not needed 
//void printShape(Matrix const& m);
//dll void TEST_(FCN* fcnPTR) {
//
//    auto x = fcnPTR->trainData.Xs[0];
//    auto y = fcnPTR->trainData.Ys[0];
//
//    //exit(0);
//
//    print(x); //print(y);
//
//    vectorList<vectorList<Matrix>> out = fcnPTR->backward(x, y);
//
//    //exit(0);
//
//    //print("weights");
//    //auto& w = out[0];
//    //for (auto e : w) {
//    //    printShape(e);
//    //}
//    print("bias");
//    for (auto e : out[1]) {
//        printShape(e);
//    }
//
//    //print("weights");
//    //auto& w1 = fcnPTR->weights;
//    //for (auto e : w1) {
//    //    printShape(e);
//    //}
//    print("bias");
//    for (auto e : fcnPTR->bias) {
//        printShape(e);
//    }
//
//
//    print("weights");
//    for (auto e : fcnPTR->weights) {
//        printShape(e);
//    }
//    for (auto e : out[0]) {
//        printShape(e);
//    }
//
//
//    exit(0);
//
//
//}
//dll int works_(int x) {
//    dll_sanity_check(x);
//    return x;
//}









