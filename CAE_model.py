import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

## Hyperparams

BATCH_SIZE = 128
NUM_EPOCHS = 100
DEVICE = 'cuda'
SAVE_PATH = "./CAE_model.pt"

## Loading in the dataset

# train
MNIST_train = torchvision.datasets.MNIST(
    root="./data", 
    train=True, 
    download=True,
    transform=transforms.ToTensor()
)
MNIST_trainloader = torch.utils.data.DataLoader(
    MNIST_train, 
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# test
MNIST_test = torchvision.datasets.MNIST(
    root="./data", 
    train=False, 
    download=True,
    transform=transforms.ToTensor()
)
MNIST_testloader = torch.utils.data.DataLoader(
    MNIST_train, 
    batch_size=BATCH_SIZE,
    shuffle=False,
)

## Creating the model
class CAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 3)
        
        # Decoder
        self.conv4 = nn.ConvTranspose2d(32, 16, 3)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv5 = nn.ConvTranspose2d(16, 1, 3)
        self.conv6 = nn.ConvTranspose2d(1, 1, 3)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)

        # Decoder
        x = self.conv4(x)
        x = F.relu(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x
    
model = CAE().to(DEVICE)

## Functions to visualize the data

dataiter = iter(MNIST_trainloader)
images, labels = next(dataiter)
plt.figure()
plt.imshow(images[0,0,:,:])
plt.title("dara visualization: this is a " + str(labels[0].item()))

## Creating a training loop

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0

    for i, data in tqdm(enumerate(MNIST_trainloader, 0)):
        # Get inputs and labels and put them on device
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zeroing gradients
        optimizer.zero_grad()

        # forward, backward, optimize
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 400 == 399:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.10f}')
            running_loss = 0.0

# saving model
torch.save(model.state_dict(), SAVE_PATH)

# printing a picture run through the model to see what the reconstruction looks like
plt.figure()
outputs = model(inputs)
plt.subplot(1,2,1)
plt.imshow(inputs[1,0,:,:].to('cpu').detach().numpy())
plt.title("Original Image")
plt.subplot(1,2,2)
plt.imshow(outputs[1,0,:,:].to('cpu').detach().numpy())
plt.title("Reconstructed Image")
plt.show()

print('Finished Training')
