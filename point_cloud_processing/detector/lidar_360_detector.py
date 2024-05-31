import torch
import torch.nn as nn

class MyLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MyLSTMClassifier, self).__init__()
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
    
    def detect(self, feature_set, model_path):
        inputs = torch.tensor(feature_set, dtype=torch.float32) 
        self.load_state_dict(torch.load(model_path))
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted
