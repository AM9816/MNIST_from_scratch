import torch
from torch import nn, Tensor as T
from torchvision.datasets import MNIST

from time import time as now

from c_init import C_FCN_Interface

from dataloader import load_or_download_MNIST

def to_pytorch_tensor(dataset):
    return\
        torch.tensor([x for x, _ in dataset], dtype=torch.float32),\
        torch.tensor([int(l) for _, l in dataset], dtype=torch.long)


def compare_with_pytorch(dllpath, threads = 4, bsize = 128, 
                         epochs = 3, lr = .05, momentum = .9):
    

    # both methods are to learn from the same dataset to keep the 
    # comparison as fair as possible
    mnist = load_or_download_MNIST()

    xTrain, yTrain = to_pytorch_tensor(mnist["train"])
    xTest, yTest = to_pytorch_tensor(mnist["test"])

    torch.set_num_threads(threads)


    # no softmax as pytorch CrossEntropyLoss requires raw logits
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10)) 

    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)
    
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=0.0)
    lossfn = nn.CrossEntropyLoss()
    
    n = xTrain.shape[0]
    startTime = now()
    
    
    for epoch in range(epochs):
        
        with torch.no_grad():
            model.eval()
            acc = float(
                (model(xTest).argmax(1) == yTest).float().mean())
            print(epoch, acc)

        model.train()

        batchIndexes = torch.randperm(n);
        for i in range(0, n, bsize):
            indexes = batchIndexes[i:i+bsize]
            
            # normal pytorch training step on tensors
            opt.zero_grad()
            out = model(xTrain[indexes])
            loss = lossfn(out, yTrain[indexes])
            loss.backward()
            opt.step()


    torch_time = now() - startTime

    model.eval()
    torch_accuracy = float(
        (model(xTest).argmax(1) == yTest).float().mean())




    # my implementation

    fcn = C_FCN_Interface(dllpath)
    fcn.create_fcn([784, 256, 128, 10], 0, 'relu')
    fcn.add_dataset(mnist, bsize, 10, True)

    startTime = now()
    fcn._train_thread(epochs, lr, momentum, 0)
    custom_time = now() - startTime

    custom_accuracy = fcn.test_accuracy()

    print(f'pytorch accuracy = {torch_accuracy} in {custom_time}s')
    print(f'custom implementation accuracy = {custom_accuracy} in {torch_time}s')

   



