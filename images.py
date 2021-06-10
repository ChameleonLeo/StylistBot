import torch

default_content_image = torch.load('images/content_img.pt')
default_style_image = torch.load('images/style_img.pt')


class Images:
    def __init__(self):
        self.content_image = 0
        self.style_image = 0

    def default_set(self):
        self.content_image = default_content_image
        self.style_image = default_style_image
