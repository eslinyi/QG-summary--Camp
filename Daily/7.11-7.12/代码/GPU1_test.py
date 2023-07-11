import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


train_data = torchvision.datasets.CIFAR10(root='./dataset',train = True,transform=torchvision.transforms.ToTensor()
                                          ,download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset',train = False,transform=torchvision.transforms.ToTensor()
                                          ,download=True)


train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度：{}".format(train_data_size))
print("测试集的长度：{}".format(test_data_size))

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,1,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,1,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10))
    def forward(self,x):
        x = self.model1(x)
        return x


train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

tudui = Tudui()
tudui = tudui.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

optimizer = torch.optim.SGD(tudui.parameters(),lr = 0.01)


total_train_step = 0

total_test_step = 0

epoch = 10

writer = SummaryWriter("./logs_train")


for i in range(epoch):
    print("---------------第{}轮的开始---------".format(i))

    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step+1
        if total_train_step %100 ==0:
            print("训练次数：{}，Loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_step + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step += 1

writer.close()