import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

import time

from app.model.model_loader import GANLoader
from app.model.image_loader import ImageLoader


GAN_WOMEN_MODEL_PATH = "./app/weights/netG_ToWomen.pth"
GAN_MEN_MODEL_PATH = "./app/weights/netG_ToMen.pth"


class ConvertFace:
    
    transform = transforms.Compose([
        transforms.Pad(padding = 1, fill = 0, padding_mode = 'symmetric'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    
class ConvertFaceToWomen(ConvertFace):
    def gen_women(self):
        

        img = Image.open(img_path)
        img = self.transform(img)
        img = img.view(1,3,img.shape[1],img.shape[2])

        fake_img = gen_Model(img).squeeze().permute(1,2,0)
        fake_img = 0.5 * (fake_img.data.numpy() + 1.0)

        return fake_img

    
class ConvertFaceToManf(ConvertFace):

    def gen_man(self):
        gen_Model = GANLoader()

        img = Image.open(img_path)
        img = self.transform(img)
        img = img.view(1,3,img.shape[1],img.shape[2])

        fake_img = gen_Model(img).squeeze().permute(1,2,0)
        fake_img = 0.5 * (fake_img.data.numpy() + 1.0)

        return fake_img


