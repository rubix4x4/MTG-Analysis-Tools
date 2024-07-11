import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Parameters
input_size = 55
hidden_size = 128
output_size = 1

# Model
model = SimpleNN(input_size, hidden_size, output_size)

# Print the model architecture
print(model)

# Loss and optimizer
criterion = nn.MSELoss()  # For regression tasks; use nn.CrossEntropyLoss for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (example with dummy data)
num_epochs = 100
for epoch in range(num_epochs):
    # Dummy inputs and targets
    inputs = torch.randn(10, input_size)
    targets = torch.randn(10, output_size)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
