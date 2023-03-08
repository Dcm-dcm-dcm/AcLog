from model import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate Dataset
def generate(name, window_size, dir):
    num_sessions = 0
    inputs = []
    outputs = []
    with open('data/' + dir + '/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


def re_training_random(rounds, dir, num_classes=28,
                       num_layers=2, hidden_size=64, window_size=10):
    # Hyperparameters
    num_epochs = 300
    batch_size = 2048
    input_size = 1
    model_dir = 'model'

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate('human_labeled_samples', window_size, dir)
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
    print('save model to' + 'model/' + dir + '/re_train_model_rounds_' + str(rounds) + '.pt')
    torch.save(model.state_dict(),
               'model/' + dir + '/re_train_model_rounds_' + str(rounds) + '.pt')
    print('Finished Training')


def re_training(start, window, dir, num_classes=28,
                num_layers=2, hidden_size=64, window_size=10, epochs=300, batch_size = 2048):
    # Hyperparameters
    num_epochs = epochs
    input_size = 1
    model_dir = 'model'

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate('human_labeled_samples', window_size, dir)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    for epoch in tqdm(range(num_epochs), desc="Training: "):  # Loop over the dataset multiple times
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
    print('save model to: ' + 'model/' + dir + '/re_train_model_start' + str(start) + '_window' + str(window) + '.pt')
    torch.save(model.state_dict(),
               'model/' + dir + '/re_train_model_start' + str(start) + '_window' + str(window) + '.pt')
    print('Finished Training')
