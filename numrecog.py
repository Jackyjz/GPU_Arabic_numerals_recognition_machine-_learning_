import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the transformation with data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),  # Random rotation
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a more complex CNN model
class DeeperConvNet(nn.Module):
    def __init__(self):
        super(DeeperConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*7*7, 128)  # Adjusted based on output size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 128*7*7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, move it to the appropriate device (GPU or CPU)
model = DeeperConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop with checkpointing
num_epochs = 10  # Number of epochs
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Time: {epoch_time:.2f} seconds')

    # Save checkpoint at the end of each epoch
    checkpoint_path = f'deeper_mnist_model_epoch_{epoch+1}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved as '{checkpoint_path}'")

print("Training complete")

# Final model evaluation on the test set
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Save the final model
final_model_path = 'deeper_mnist_model_final.pth'
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved as '{final_model_path}'")

# Load and test the model with a new image
model = DeeperConvNet().to(device)
model.load_state_dict(torch.load(final_model_path))
model.eval()

# Load and preprocess the image
image_path = r"C:\Users\jacky\Downloads\NUM_TEST.png"  # Replace with the correct path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))

# Invert if necessary (MNIST digits are white on black background)
image = cv2.bitwise_not(image)

# Normalize the image
image = image.astype(np.float32) / 255.0
image = (image - 0.5) / 0.5

# Convert the image to a tensor and add a batch dimension
image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)  # Move the tensor to the device

# Perform the prediction
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    predicted_digit = predicted.item()

print(f'The model predicts this digit is: {predicted_digit}')

# Visualize the image and prediction
plt.imshow(image, cmap='gray')
plt.title(f"Model predicted: {predicted_digit}")
plt.show()
