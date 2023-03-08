from model import *

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate Dataset
def generate(name, window_size):
    dataset = []
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            dataset.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(dataset)))
    return dataset


def predict(
        dir,
        num_classes=1145,
        num_layers=2,
        hidden_size=64,
        window_size=3,
        num_candidates=114,
        threshold=25
):
    # Hyper parameters
    input_size = 1
    model_name = 'pre_train'
    model_dir = 'model/' + dir
    model_path = model_dir + '/' + model_name + '.pt'
    test_normal = './data/' + dir + '/test_normal'
    test_abnormal = './data/' + dir + '/test_abnormal'

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = generate(test_normal, window_size)
    test_abnormal_loader = generate(test_abnormal, window_size)
    fp_seq = []
    tp_seq = []
    TP = 0
    FP = 0
    # Test the model
    with torch.no_grad():
        for line in test_normal_loader:
            anomaly_num = 0
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    anomaly_num += 1
                    if anomaly_num >= threshold:
                        FP += 1
                        fp_seq.append(line[i:i + window_size + 1])
                        fp_seq.append(line)
                        break
    with torch.no_grad():
        for line in test_abnormal_loader:
            anomaly_num = 0
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    anomaly_num += 1
                    if anomaly_num >= threshold:
                        TP += 1
                        tp_seq.append(line)
                        break
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / max((TP + FP), 1)
    R = 100 * TP / max((TP + FN), 1)
    F1 = 2 * P * R / (P + R)
    print(
        'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
            FP, FN, P, R, F1))
    print('Finished Predicting')
