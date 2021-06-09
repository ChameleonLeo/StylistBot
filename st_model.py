from io import BytesIO
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_loader(image_name):
    loader = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class ContentLoss(nn.Module):
    # среднеквадратичная ошибка контента input'а и target'а
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = func.mse_loss(self.target, self.target)

    def forward(self, inp):
        self.loss = func.mse_loss(inp, self.target)
        return inp


def gram_matrix(inp):
    batch_size, h, w, f_map_num = inp.size()
    features = inp.view(batch_size * h, w * f_map_num)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):
    # среднеквадратичная ошибка стиля input'а и target'а
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = func.mse_loss(self.target, self.target)

    def forward(self, inp):
        gram = gram_matrix(inp)
        self.loss = func.mse_loss(gram, self.target)
        return inp


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn = torch.load('vgg/vgg19.pth').eval()


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(content_img, style_img,
                               conv_net,
                               normalization_mean,
                               normalization_std):
    conv_net = copy.deepcopy(conv_net)

    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in conv_net.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers_default:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers_default:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(content_img, style_img, input_img,
                       conv_net=cnn,
                       normalization_mean=cnn_normalization_mean,
                       normalization_std=cnn_normalization_std,
                       num_steps=500,
                       style_weight=100000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(content_img,
                                                                     style_img,
                                                                     conv_net,
                                                                     normalization_mean,
                                                                     normalization_std)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    output_img = np.rollaxis(input_img.detach().numpy()[0], 0, 3)
    result = Image.fromarray(np.uint8(output_img * 255))
    to_bytes = BytesIO()
    result.save(to_bytes, 'PNG')
    result = to_bytes.seek(0)

    return result
