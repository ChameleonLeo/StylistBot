import os
import random

from main import bot


class Images:
    def __init__(self):
        self.content_image = 0
        self.style_image = 0


images = {}


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


async def set_random_default_set():
    content_name = random.choice(os.listdir("images/content"))
    style_name = random.choice(os.listdir("images/style"))
    content_image = open(f'images/content/{content_name}', 'rb')
    style_image = open(f'images/style/{style_name}', 'rb')
    return content_image, style_image
