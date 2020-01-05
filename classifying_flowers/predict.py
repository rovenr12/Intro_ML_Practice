import argparse
import torch
from torchvision import models
from torch import nn
from collections import OrderedDict
import numpy as np
from PIL import Image
import json

parser = argparse.ArgumentParser(description='Predict image')
parser.add_argument('image_path', type=str, help='Load Image')
parser.add_argument('checkpoint', type=str, help='Load Model')
parser.add_argument('--top_k', type=str, default=5, help='Return top KK most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Use a mapping of categories')
parser.add_argument('--gpu', help='Use GPU to train', action='store_true')

args = parser.parse_args()


# Load Pre-trained model function
def switch_model(argument):

    if argument == 'Alexnet':
        pre_train_model = models.alexnet(pretrained=True)
        input_nodes= pre_train_model.classifier[1].in_features
        print("Using Alexnet")
    elif argument == 'Densent':
        pre_train_model = models.densenet161(pretrained=True)
        input_nodes= pre_train_model.classifier.in_features
        print("Using Densent")
    elif argument == 'Mobilenet':
        pre_train_model = models.mobilenet_v2(pretrained=True)
        input_nodes= pre_train_model.classifier[1].in_features
        print("Using Mobilenet")
    elif argument == 'Mnasnet':
        pre_train_model = models.mnasnet1_0(pretrained=True)
        input_nodes= pre_train_model.classifier[1].in_features
        print("Using Mnasnet")
    else:
        pre_train_model = models.vgg16(pretrained=True)
        input_nodes= pre_train_model.classifier[0].in_features
        print("Using Vgg")

    return pre_train_model, input_nodes


# Load Model from checkpoint.pth
def load_model(file_path):

    data = torch.load(file_path)
    model, input_nodes = switch_model(data.get('arch'))
    hidden_nodes = data.get('hidden_units')

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_nodes, hidden_nodes)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p=0.4)),
        ('fc2', nn.Linear(hidden_nodes, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = torch.device("cpu")

    return model, device


# Process Image
def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """

    img = image.resize((224, 224))
    img = np.array(img)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img / 255 - mean) / std
    img = img.transpose((2, 0, 1))

    return img


with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


# Predict
model, device = load_model(args.checkpoint)
image = Image.open(args.image_path)
image = process_image(image).reshape((1,) + process_image(image).shape)
image = torch.from_numpy(image)
image = image.type(torch.FloatTensor).to(device)

model.eval()
with torch.no_grad():
    output = model.forward(image)

ps = torch.exp(output).cpu()
top_p, top_class = ps.topk(args.top_k, dim=1)
top_class_name = []
probs = top_p[0].detach().numpy()

for i in range(5):
    name = top_class[0][i].numpy()
    name += 1
    top_class_name.append(cat_to_name.get(str(name)))

print(probs)
print(top_class_name)