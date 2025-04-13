import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Define directories and load data
result_folder = "/media/yi/KESU/anti_uav/result"
save_directory = os.path.join(result_folder, "lidar_360_feature_set")

feature_set_train = np.load(os.path.join(save_directory, 'feature_train.npy'))
label_set_train = np.load(os.path.join(save_directory, 'label_train.npy'))
feature_set_test = np.load(os.path.join(save_directory, 'feature_val.npy'))
label_set_test = np.load(os.path.join(save_directory, 'label_val.npy'))

# Filter data based on labels
ind = label_set_train == 1
gt_cluster_train = feature_set_train[ind]

ind = label_set_test == 1
gt_cluster_test = feature_set_test[ind]
ind = label_set_test == 0
bg_cluster_test = feature_set_test[ind]

# Function to reverse the sequence
def reverse_sequence(data):
    return torch.flip(data, dims=[1])

# Function to randomly replace features with all zeros
def random_replace_with_zeros(data, max_replace=3):
    seq_len, num_features = data.size()
    replace_count = np.random.randint(1, max_replace+1)  # Randomly choose number of timestamps to replace
    
    indices = np.random.choice(seq_len, replace_count, replace=False)  # Randomly choose timestamps to replace
    data[indices, :] = 0
    
    return data

# Augment the data
def augment_data(X_tensor, apply_augmentation=True):
    if not apply_augmentation:
        return X_tensor
    
    X_augmented = []
    for x in X_tensor:
        # Randomly choose augmentation technique
        augmentation = np.random.choice(['original', 'reverse', 'random_replace'], p=[0.5, 0.25, 0.25])
        if augmentation == 'reverse':
            x_augmented = reverse_sequence(x)
        elif augmentation == 'random_replace':
            x_augmented = random_replace_with_zeros(x)
        else:
            x_augmented = x
        X_augmented.append(x_augmented)
    
    X_augmented = torch.stack(X_augmented)
    
    return X_augmented

# Convert data and labels to PyTorch tensors
X_tensor_train = torch.tensor(feature_set_train, dtype=torch.float32) 
y_tensor_train = torch.tensor(label_set_train, dtype=torch.long)

X_tensor_test = torch.tensor(feature_set_test, dtype=torch.float32)  
y_tensor_test = torch.tensor(label_set_test, dtype=torch.long) 

# Augment the training data
X_tensor_train = augment_data(X_tensor_train, apply_augmentation=True)

# Define batch size
batch_size = 64

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_tensor_test, y_tensor_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize model
input_size = 9  # Number of features
hidden_size = 64
num_layers = 1
num_classes = 2  # Number of unique classes
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
model_path = 'lstm_model.pth'
best_valid_loss = float('inf')
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = augment_data(inputs, apply_augmentation=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct / total
    
    # Evaluating the model
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    
    if test_loss < best_valid_loss:
        best_valid_loss = test_loss
        torch.save(model.state_dict(), model_path)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

# Load the best model
model.load_state_dict(torch.load(model_path))

# Evaluate the model on the test set
model.eval()
running_loss = 0.0
correct = 0
total = 0
true_positive = 0
false_positive = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        true_positive += ((predicted == 1) & (labels == 1)).sum().item()
        false_positive += ((predicted == 0) & (labels == 1)).sum().item()
    
test_loss = running_loss / len(test_loader.dataset)
test_accuracy = correct / total
test_recall = true_positive / (true_positive + false_positive)
print(f'Test Acc: {test_accuracy:.4f}, Test Recall: {test_recall:.4f}')
