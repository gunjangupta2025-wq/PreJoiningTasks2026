import numpy as np

class Activations:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # clip to avoid overflow
    
    @staticmethod
    def sigmoid_der(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_der(x):
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_der(x):
        return 1.0 - np.tanh(x)**2

class NeuralNetworkFromScratch:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        # Initialize weights using He (MSRA) initialization for ReLU, or Xavier for others
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
        # Set activation functions
        if activation == 'relu':
            self.act, self.act_der = Activations.relu, Activations.relu_der
        elif activation == 'tanh':
            self.act, self.act_der = Activations.tanh, Activations.tanh_der
        else:
            self.act, self.act_der = Activations.sigmoid, Activations.sigmoid_der

    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.act(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = Activations.sigmoid(self.Z2) # Assuming binary classification output
        return self.A2

    def backward(self, Y):
        m = Y.shape[0]
        
        # Loss derivative (MSE Loss assumed here for simple gradient tracking)
        # dL/dA2 = (A2 - Y)
        dZ2 = (self.A2 - Y) / m 
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Backprop into hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.act_der(self.Z1)
        dW1 = np.dot(self.X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2

def train(self, X, Y, epochs=100, batch_size=32, lr=0.01):
        m = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data every epoch
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # Forward & Backward pass
                self.forward(X_batch)
                dW1, db1, dW2, db2 = self.backward(Y_batch)
                
                # Update parameters
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db
                
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Build Network using torch.nn.Module
class PyTorchCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PyTorchCNN, self).__init__()
        
        # 2. CNN Layers
        self.features = nn.Sequential(
            # Input: (batch, 1, 28, 28) for MNIST-like data
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (batch, 16, 14, 14)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # Output: (batch, 32, 7, 7)
        )
        
        # Fully Connected Layers with Regularization (Dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Regularization: Dropout
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. Create Custom Data Loader Pipeline
class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Create random image-like data: (Samples, Channels, Height, Width)
        self.X = torch.randn(num_samples, 1, 28, 28)
        self.Y = torch.randint(0, 10, (num_samples,))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Initialize Pipeline elements
dataset = SyntheticImageDataset()
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PyTorchCNN(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

# 4. Regularization: Added weight_decay for L2 penalty
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# 5. Learning Rate Scheduling: Drop LR by 0.1 every 5 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + Backward + Optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    # Step the learning rate scheduler
    scheduler.step()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Current LR: {current_lr}")