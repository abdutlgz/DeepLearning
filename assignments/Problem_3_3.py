import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dl.models import LeNet5

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 40

# Dataset preparation
transform = transforms.ToTensor()

# Download and load QMNIST dataset
train_dataset = torchvision.datasets.QMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.QMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
model = LeNet5()

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training function
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
# Testing function
def test_model(model, test_loader, criterion):
    model.eval()  # Evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_test_loss, accuracy

# Main training loop
if __name__ == "__main__":
    # Train the model
    train_model(model, train_loader, optimizer, criterion, epochs)
    
    # Test the model
    test_model(model, test_loader, criterion)
    
    # Save the model for later reuse
    torch.save(model.state_dict(), 'lenet5_qmnist.pth')
    print('Model saved as lenet5_qmnist.pth')
