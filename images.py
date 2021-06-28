import os
import random
from main import bot

import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import numpy as np
import torch


class Images:
    def __init__(self):
        self.content_image = 0
        self.style_image = 0


# Dict for handling user's images (set from default or received)
images = {}


# Func for retrieving the path of received file
async def get_file_path(message_info):
    if message_info.content_type == 'photo':
        file_info = message_info.photo[-1]
    elif message_info.content_type == 'document':
        file_info = message_info.document
    else:
        return False
    file_id = await bot.get_file(file_info.file_id)
    file_path = await bot.download_file(file_id.file_path)
    return file_path


# Func for choosing random set of images
async def set_random_default_set():
    content_name = random.choice(os.listdir("images/content"))
    style_name = random.choice(os.listdir("images/style"))
    content_image = open(f'images/content/{content_name}', 'rb')
    style_image = open(f'images/style/{style_name}', 'rb')
    return content_image, style_image


# Func for transformating image to tensor
async def image_loader(image_name):
    loader = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)


# Func for transformating output tensor to bytes for sending an image
async def to_bytes(input_img):
    output_img = np.rollaxis(input_img.detach().numpy()[0], 0, 3)
    result = Image.fromarray(np.uint8(output_img * 255))
    result_to_bytes = BytesIO()
    result.save(result_to_bytes, 'JPEG')
    result_to_bytes.seek(0)
    return result_to_bytes
