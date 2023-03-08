from model import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate Dataset
def generate(name, window_size):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            line = line + [-1] * (window_size + 1 - len(line))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


def train(
        training_file, model_name, model_dir,
        num_classes=28, num_layers=2, hidden_size=64, window_size=10,
        num_epochs=300
):
    # Hyperparameters
    batch_size = 2048
    input_size = 1

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate(training_file, window_size)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + model_name + '.pt')
    print('Finished Training')
