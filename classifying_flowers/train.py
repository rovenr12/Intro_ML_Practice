import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Training the DNN for flowers')
parser.add_argument('data_directory', type=str, help='Data Dictionary')
parser.add_argument('--save_dir', type=str, help='Set directory to save checkpoints')
parser.add_argument('--arch', type=str, default='VGG', help='Choose architecture(Default: VGG)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Set hyperparameter(Learning rate)')
parser.add_argument('--hidden_units', type=int, default=512, help='Set hyperparameter(Hidden units)')
parser.add_argument('--epochs', type=int, default=10, help='Set hyperparameter(Epoch)')
parser.add_argument('--gpu', help='Use GPU to train', action='store_true')
parser.add_argument('--show_architecture', help='Show available architecture', action='store_true')

args = parser.parse_args()

if args.show_architecture:
    print("The List of architecture (5 choices): .. "
          "1. VGG .."
          "2. Alexnet .."
          "3. Densent .."
          "4. Mobilenet .."
          "5. Mnasnet ..")

# Load data
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Load the datasets
train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

# Using the datasets and the trainforms, define the loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)


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


# Creat pre-trained model
model, input_nodes = switch_model(args.arch)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_nodes, args.hidden_units)),
    ('relu', nn.ReLU()),
    ('drop', nn.Dropout(p=0.4)),
    ('fc2', nn.Linear(args.hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
else:
    device = torch.device("cpu")


#  Calculate the loss and accuracy
def accuracy_test(loader):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get Output & Calculate the lose
            output = model.forward(inputs)
            loss = criterion(output, labels)

            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    model.train()
    return test_loss, accuracy


epochs = args.epochs

for epoch in range(epochs):

    running_loss = 0

    # Train
    for inputs, labels in trainloader:

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    test_loss, accuracy = accuracy_test(validloader)
    print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / len(trainloader):.3f}.. "
          f"Validation loss: {test_loss / len(validloader):.3f}.. "
          f"Validation accuracy: {accuracy / len(validloader):.3f}")

# Save models
checkpoint = {'state_dict': model.state_dict(),
              'arch': args.arch,
              'hidden_units': args.hidden_units}

if args.save_dir is not None:
    file_name = args.save_dir + "/" + args.arch + ".pth"
    torch.save(checkpoint, file_name)
    print("Your model is saved in {} and name is {}.pth".format(args.save_dir, args.arch))

else:
    file_name = args.arch + ".pth"
    torch.save(checkpoint, file_name)
    print("Your model is saved and the name is {}.pth".format(args.arch))