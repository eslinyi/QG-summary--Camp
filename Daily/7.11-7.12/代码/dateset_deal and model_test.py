import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dateset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dateset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dateset_transform,download=True)

# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# writer = SummaryWriter("p10")
# for i in range(20):
#     img,target = test_set[i]
#     writer.add_image("test_set",img,i)
#
# writer.close()

dateloader = DataLoader(test_set,batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10))
    def forward(self,x):
        x = self.model1(x)
        return x

Loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(params=tudui.parameters(),lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dateloader:
        imgs,targets = data
        outputs = tudui(imgs)
        result_loss = Loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)