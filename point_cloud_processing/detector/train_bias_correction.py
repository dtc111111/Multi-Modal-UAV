import torch
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Load the data
mat_data = sio.loadmat('depth_correction_filtered.mat')

# Prepare the input and target data
X = torch.tensor(mat_data['filtered_data_matrix_360'][:, 1:4], dtype=torch.float32)
Y = torch.tensor(mat_data['filtered_data_matrix_360'][:, 4:], dtype=torch.float32)

# Split the data into training, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
total_samples = len(X)

train_samples = int(train_ratio * total_samples)
val_samples = int(val_ratio * total_samples)
test_samples = total_samples - train_samples - val_samples

train_data = TensorDataset(X[:train_samples], Y[:train_samples])
val_data = TensorDataset(X[train_samples:train_samples + val_samples], Y[train_samples:train_samples + val_samples])
test_data = TensorDataset(X[train_samples + val_samples:], Y[train_samples + val_samples:])

# Create DataLoader for training, validation, and test sets
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_data)}, Validation Loss: {val_loss / len(val_data)}")

# Test the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)

print(f"Test Loss: {test_loss / len(test_data)}")

# Save the model
torch.save(model.state_dict(), 'depth_correction.pth')
