import pandas as pd
from random import random
from Levenshtein import distance
from tqdm import tqdm


def generate(name, window_size, dir):
    dataset = []
    with open('data/' + dir + '/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            for i in range(len(ln) - window_size):
                dataset.append(ln[i:i + window_size + 1])
    print('Number of seqs({}): {}'.format(name, len(dataset)))

    return dataset


def write_to_samples(line, fp):
    for item in line:
        if item != -1:
            fp.write(str(item) + ' ')
    fp.write('\n')


def isValid(normal_seq, abnormal_seq, thre):
    normal_seq_str = ''
    abnormal_seq_str = ''
    for seq in normal_seq:
        normal_seq_str += str(seq)
    for seq in abnormal_seq:
        abnormal_seq_str += str(seq)
    # drop if the last bit is the same and the log sequence is similar
    if (distance(normal_seq_str, abnormal_seq_str) <= thre) \
            and (normal_seq_str[len(normal_seq_str) - 1] == abnormal_seq_str[len(abnormal_seq_str) - 1]):
        return False
    else:
        return True


def merging(dir, levenshtein=False, thre=2, drop_out=1, window_size=10):
    filename = './data/' + dir + '/selected_samples'
    dis_filename = './data/' + dir + '/human_labeled_samples'
    if levenshtein:
        dis_file = open(dis_filename, 'w')
        print('merging files, thre=' + str(thre) + ', drop_out=' + str(drop_out))

        struct_data = pd.read_csv(filename, engine='c',
                                  na_filter=False, memory_map=True)
        normal_seq_list = []
        abnormal_seq_list = []
        sum = 0
        for idx, row in struct_data.iterrows():
            line = row['seq']
            label = row['label']
            ln = list(map(lambda n: n, map(int, line.strip().split())))
            if label == 0:  # normal
                normal_seq_list.append(ln)
            else:  # abnormal
                abnormal_seq_list.append(ln)

        for normal_seq in normal_seq_list:  # Normal samples are retained
            write_to_samples(normal_seq, dis_file)
            sum += 1

        # Filter the original training set
        selected_sum = 0
        raw_normal_seq_list = generate('labeled_samples', window_size, dir)
        for raw_normal_seq in tqdm(raw_normal_seq_list, desc="Filtering: "):
            flag = True
            for abnormal_seq in abnormal_seq_list:
                if not isValid(raw_normal_seq, abnormal_seq, thre):  # Judge whether it is legal
                    flag = False
                    break
            if flag:
                write_to_samples(raw_normal_seq, dis_file)
                sum += 1
                selected_sum += 1
            else:  # random drop out
                ran = random()
                if ran > drop_out:
                    write_to_samples(raw_normal_seq, dis_file)
                    sum += 1
                    selected_sum += 1
        print('selected_seqs_from_raw_dataset: ' + str(selected_sum))
        print('all_seqs: ' + str(sum))
        print('close files')
        dis_file.close()
    else:
        dis_file = open(dis_filename, 'w')
        print('merging files')

        struct_data = pd.read_csv(filename, engine='c',
                                  na_filter=False, memory_map=True)
        normal_seq_list = []
        abnormal_seq_list = []
        sum = 0
        for idx, row in struct_data.iterrows():
            line = row['seq']
            label = row['label']
            ln = list(map(lambda n: n, map(int, line.strip().split())))
            if label == 0:  # normal
                normal_seq_list.append(ln)
            else:  # abnormal
                abnormal_seq_list.append(ln)

        for normal_seq in normal_seq_list:  # Normal samples screened manually are retained
            write_to_samples(normal_seq, dis_file)
            sum += 1
        print('close files')
        dis_file.close()
