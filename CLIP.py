import os
import clip
import torch
from torchvision.datasets import CIFAR10
from PIL import Image

# Load the model
device = "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)


def run(path):
    # Prepare the inputs
    image = Image.open(path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    # Print the result
    for value, index in zip(values, indices):
        print(f"{cifar10.classes[index]:>16s}: {100 * value.item():.2f}%")

dir="C:/Users/ameli/Documents/EMC/Images"
for name in os.listdir(dir):
    print(f"Image '{name.split('.')[0]}'")
    path = dir+"/"+name
    run(path)
    print()