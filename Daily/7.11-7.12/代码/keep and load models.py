import torchvision
import torch

##keep1
# vgg16 = torchvision.models.vgg16(pretrained = False)
# torch.save(vgg16,"vgg16_method1.pth")

##load1
# model = torch.load("vgg16_method1.pth")
# print(model)

##keep2
# torch.save(vgg16.state_dict(),"vgg16_method2.pth")

##load2
vgg16 = torchvision.models.vgg16(pretrained = False)
# model = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
