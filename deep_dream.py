import sys
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import resize

seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self, pretrained_model, feature_layers):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.feature_layers = feature_layers

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.pretrained_model.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
            if i == self.feature_layers[-1]:
                return features


def ascend(image, model, lower_clamp, upper_clamp, step_size=0.05, jitter=32):
    with torch.no_grad():
        shift_y, shift_x = torch.randint(-jitter, jitter + 1, (2,))
        image = torch.roll(image, shifts=(shift_y, shift_x), dims=(2, 3))
        image.requires_grad = True

    L = 0
    features = model(image)
    for feature in features:
        L += torch.mean(feature**2)
    L /= len(features)
    L.backward()

    with torch.no_grad():
        image += step_size / torch.mean(torch.abs(image.grad)) * image.grad
        image.grad = None

        image = torch.roll(image, shifts=(-shift_y, -shift_x), dims=(2, 3))
        image.clamp_(lower_clamp, upper_clamp)

    return image


def deepdream(image, model, preprocess, epochs=10, octaves=4, octave_zoom=1.4):
    p_std = torch.tensor(preprocess.std)
    p_mean = torch.tensor(preprocess.mean)
    p_std = torch.reshape(p_std, (1, 3, 1, 1)).to(device)
    p_mean = torch.reshape(p_mean, (1, 3, 1, 1)).to(device)

    lower_clamp = -p_mean / p_std
    upper_clamp = (1 - p_mean) / p_std

    image_octaves = [preprocess(image)]
    for i in range(octaves):
        sh, sw = image_octaves[-1].shape[2:4]
        sh = int(sh // octave_zoom)
        sw = int(sw // octave_zoom)
        image_octave = resize(image_octaves[-1], (sh, sw))
        image_octaves.append(image_octave)

    detail = torch.zeros(image_octaves[-1].shape)
    for i, image_octave in enumerate(reversed(image_octaves)):
        if i > 0:
            sh, sw = image_octave.shape[2:4]
            detail = resize(detail, (sh, sw))

        input = image_octave + detail
        pbar = tqdm(range(1, epochs + 1))
        for epoch in pbar:
            input = ascend(input, model, lower_clamp, upper_clamp)
            pbar.set_postfix({"Octave": f"{i + 1}/{octaves + 1}"})

        detail = input - image_octave

    image = input
    image = image * p_std + p_mean
    image = torch.clamp(image, 0, 1)

    return image


def dream(random=False):
    if random:
        image = torch.rand((1, 3, 300, 300))
        image_name = "Noise.jpg"
    else:
        image_name = sys.argv[1]
        image_path = os.path.abspath(image_name)
        image = read_image(image_path).float() / 255
        image = image.unsqueeze(0)
        if image.shape[1] >= 3:
            image = image[:, 0:3, :, :]
        image = image.to(device)

    image_height, image_width = image.shape[2:4]

    pretrained_weights = VGG19_Weights.DEFAULT
    pretrained_model = vgg19(weights=pretrained_weights)

    # feature_layers = [5, 10, 19, 28]
    feature_layers = [28]
    model = Model(pretrained_model, feature_layers)

    preprocess = pretrained_weights.transforms()
    preprocess.crop_size = [image_height, image_width]
    preprocess.resize_size = [image_height, image_width]

    image = deepdream(image, model, preprocess)

    output_file_path = os.path.join(os.getcwd(), "DreamOutput")
    if not os.path.isdir(output_file_path):
        os.mkdir(output_file_path)

    image_name = os.path.split(image_name)[-1]
    fp = os.path.join(output_file_path, image_name)
    save_image(image, fp)


def main():
    if len(sys.argv) == 2:
        dream()
    else:
        print("Argument error.")
        print("Dream image by running:")
        print(f"python {sys.argv[0]} <image>")


if __name__ == "__main__":
    main()
