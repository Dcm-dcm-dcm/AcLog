import os
from model import *
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate Dataset
def generate(name, window_size, dir):
    dataset = []
    seqs_num = 0
    with open('data/' + dir + '/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            seqs_num += (len(ln) - window_size)
            ln = ln + [-1] * (window_size + 1 - len(ln))
            dataset.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(dataset)))
    print('Number of seqs({}): {}'.format(name, seqs_num))

    return dataset


def write_to_samples(line, dir):
    filePath = './data/' + dir + '/selected_samples'
    fp = open(filePath, "a")
    for item in line:
        fp.write(str(item + 1) + ' ')
    fp.write('\n')


def write_to_samples_with_label(line, dir, label):
    filePath = './data/' + dir + '/selected_samples'
    if not os.path.exists(filePath):
        fp = open(filePath, "a")
        fp.write("seq,label\n")
    else:
        fp = open(filePath, "a")
    for item in line:
        fp.write(str(item - 1) + ' ')
    fp.write(',' + str(label) + '\n')


def select_samples_run(
        start, window, dir,
        window_size=7,
        num_classes=1145,
        levenshtein=False
):
    # Hyper parameters

    input_size = 1
    num_layers = 2
    hidden_size = 64
    model_path = 'model/' + dir + '/pre_train.pt'

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))

    # remove file
    if os.path.exists('./data/' + dir + '/selected_samples'):
        os.remove('./data/' + dir + '/selected_samples')

    # Select samples
    sum = 0
    normal_seqs = 0
    abnormal_seqs = 0
    if not levenshtein:
        unlabeled_samples_loader = generate('unlabeled_samples', window_size, dir)
        with torch.no_grad():
            for line in tqdm(unlabeled_samples_loader, desc="Select Samples: "):
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    length = int(len(torch.argsort(output, 1)[0]))
                    doubtful_labels = torch.argsort(output, 1)[0][length - start - window:length - start]
                    if label in doubtful_labels:
                        sum += 1
                        write_to_samples(line[i:i + window_size + 1], dir)
    else:
        unlabeled_samples_normal_loader = generate('unlabeled_samples_normal', window_size, dir)
        unlabeled_samples_abnormal_loader = generate('unlabeled_samples_abnormal', window_size, dir)
        with torch.no_grad():
            for line in tqdm(unlabeled_samples_normal_loader, desc="Select Samples: "):
                seq_flag = False
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    length = int(len(torch.argsort(output, 1)[0]))
                    doubtful_labels = torch.argsort(output, 1)[0][length - start - window:length - start]
                    if label in doubtful_labels:
                        sum += 1
                        seq_flag = True
                        write_to_samples_with_label(line[i:i + window_size + 1], dir, 0)
                if seq_flag:
                    normal_seqs += 1
            for line in tqdm(unlabeled_samples_abnormal_loader, desc="Select Samples: "):
                seq_flag = False
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    length = int(len(torch.argsort(output, 1)[0]))
                    doubtful_labels = torch.argsort(output, 1)[0][length - start - window:length - start]
                    if label in doubtful_labels:
                        sum += 1
                        seq_flag = True
                        write_to_samples_with_label(line[i:i + window_size + 1], dir, 1)
                if seq_flag:
                    abnormal_seqs += 1

    if sum == 0:
        if levenshtein:
            filePath = './data/' + dir + '/selected_samples'
            fp = open(filePath, "a")
            fp.write("seq,label\n")
        else:
            write_to_samples([], dir)
    if levenshtein:
        print('selected_normal_seqs: ' + str(normal_seqs))
        print('selected_abnormal_seqs: ' + str(abnormal_seqs))
    else:
        print('selected_seqs: ' + str(sum))
