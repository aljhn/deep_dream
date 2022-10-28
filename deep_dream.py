import sys
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import vgg19, VGG19_Weights
from torchvision.io import read_image
from torchvision.utils import save_image

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
                break
        return features


def dream():
    image_name = sys.argv[1]
    image_path = os.path.abspath(image_name)
    image = read_image(image_path).float() / 255
    image = image.unsqueeze(0)
    # image = torch.rand(image.shape)
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

    p_std = torch.tensor(preprocess.std)
    p_mean = torch.tensor(preprocess.mean)
    p_std = torch.reshape(p_std, (1, 3, 1, 1)).to(device)
    p_mean = torch.reshape(p_mean, (1, 3, 1, 1)).to(device)

    upper_clamp = (1 - p_mean) / p_std
    lower_clamp = -p_mean / p_std

    image = preprocess(image)
    image.requires_grad = True

    jitter = 32
    step_size = 0.2

    optimizer = Adam([image], lr=step_size)

    epochs = 10
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        shift_y, shift_x = torch.randint(-jitter, jitter + 1, (2,))
        # with torch.no_grad():
            # image = torch.roll(image, shifts=(shift_y, shift_x), dims=(2, 3))
            # image.requires_grad = True

        L = 0
        features = model(image)
        for feature in features:
            L -= torch.mean(feature**2)
        L.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            # image = torch.roll(image, shifts=(-shift_y, -shift_x), dims=(2, 3))
            image.clamp_(lower_clamp, upper_clamp)

        pbar.set_postfix({"L": L.item()})

    # Deprocess
    image = image * p_std + p_mean
    image = torch.clamp(image, 0, 1)

    output_file_path = os.path.join(os.getcwd(), "DreamOutput")
    if not os.path.isdir(output_file_path):
        os.mkdir(output_file_path)

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



# TODO:
# 1. enhance loaded images
# 1.5. try different layer activations for different types of features
# 2. send in random noise, zoom in and scale, send the zoomed in image and repeat to create an animation
# 2.5. also rotate in the zoom
# 3. send in a random path in latent space that loops around to itself to create a looping animation
# 4. random noise, translate and scale, send translate in, repeat
# 4.5. translate in random direction every iteration
# 4.6. also rotate the translation
# 5. combine zooming, translation and rotation