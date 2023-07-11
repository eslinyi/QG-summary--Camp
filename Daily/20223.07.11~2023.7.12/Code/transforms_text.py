from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
image = Image.open("111.jpg")
print(image)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(image)
writer.add_image("ToTensor",img_tensor)
writer.close()