import asyncio
import copy

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from images import to_bytes, image_loader


# MSE loss of content (target and content images)
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = func.mse_loss(self.target, self.target)

    def forward(self, content):
        self.loss = func.mse_loss(content, self.target)
        return content


# Gram matrix for computing style loss
def gram_matrix(image):
    batch_size, h, w, f_map_num = image.size()
    features = image.view(batch_size * h, w * f_map_num)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * h * w * f_map_num)


# MSE loss of style (target and style images)
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = func.mse_loss(self.target, self.target)

    def forward(self, style):
        gram = gram_matrix(style)
        self.loss = func.mse_loss(gram, self.target)
        return style


# Normalization of images (for transforming according to vgg indexes)
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# Function to compose Generating model (for creating output image)
async def compose_model(content_img, style_img, vgg):
    normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    normalization_std = torch.tensor([0.229, 0.224, 0.225])
    generating_model = nn.Sequential(Normalization(normalization_mean, normalization_std))

    conv_net = copy.deepcopy(vgg)
    layers = ['conv0', 'conv1', 'conv2', 'conv3', 'conv4']
    content_losses, style_losses = [], []
    conv = 0
    for layer in conv_net.children():
        if isinstance(layer, nn.Conv2d):
            conv += 1
            name = 'conv{}'.format(conv)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}'.format(conv)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool{}'.format(conv)
            layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        generating_model.add_module(name, layer)
        # Adding loss classes after appropriate convolutional layers
        if name == 'conv3':
            target = generating_model(content_img).detach()
            content_loss = ContentLoss(target)
            generating_model.add_module("content_loss{}".format(conv), content_loss)
            content_losses.append(content_loss)
        if name in layers:
            target_feature = generating_model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            generating_model.add_module("style_loss{}".format(conv), style_loss)
            style_losses.append(style_loss)

    return generating_model, style_losses, content_losses


# The main training process:
# - features extraction from provided images by vgg19;
# - generating of output "styled" image by composed model.
async def transferring(content_img, style_img, num_steps=100,
                       style_weight=100000, content_weight=1):
    content_img = await image_loader(content_img)
    style_img = await image_loader(style_img)
    input_img = content_img.clone()

    pretrained_net = torch.load('vgg/vgg19_cutted_5l.pth', map_location='cpu').eval()
    gen_model, style_losses, content_losses = await compose_model(content_img, style_img,
                                                                  pretrained_net)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    # Initializing of vars for saving best output (with minimal loss)
    best_score, best_content_score = 1e3, 1e3
    best_output, redefine = input_img.clone(), False

    print('Computing losses...')
    run = [0]
    while run[0] <= num_steps:
        await asyncio.sleep(0)

        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            gen_model(input_img)

            style_score = 0
            global content_score
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
            if run[0] % 20 == 0:
                print(f"Step {run[0]}: style loss {style_score:.4f}, "
                      f"content loss {content_score:.4f}\n")

            return style_score + content_score

        total_loss = closure()
        # Check score: if current loss is lower then best, redefine output image
        if (best_score > total_loss) and (best_content_score > content_score):
            best_output = input_img.clone()
            best_score = total_loss
            best_content_score = content_score
            best_output_step = run[0]
            # Redefine only if trained (to avoid redefining at first steps)
            if best_output_step > 50:
                redefine = True

        optimizer.step(closure)

    print("Final modifications of styling...")
    # If score at the last step worse then best, return output image with the best score
    if redefine:
        output = best_output.data.clamp_(0, 1)
        print(f"Image redefined, best score was at step {best_output_step}")
    else:
        output = input_img.data.clamp_(0, 1)
    output_to_bytes = await to_bytes(output)

    return output_to_bytes
