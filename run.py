from select_samples import *
from merge import *
from re_train import *
from test import *
from pre_train import *
import argparse


def running(start, window, dir, num_classes, num_candidates, threshold,
            thre, window_size, drop_out, epochs, batch_size):
    print('```````````````````````````````````````')
    print('start=' + str(start) + ',window=' + str(window))
    select_samples_run(start, window, dir, num_classes=num_classes, window_size=window_size, levenshtein=True)
    merging(dir, levenshtein=True, thre=thre, drop_out=drop_out, window_size=window_size)
    re_training(start, window, dir, window_size=window_size, num_classes=num_classes, epochs=epochs,
                batch_size=batch_size)
    test_model(start, window, dir, num_candidates=num_candidates,
               threshold=threshold, num_classes=num_classes, window_size=window_size)
    print('```````````````````````````````````````')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default='HDFS')
    parser.add_argument('-num_classes', type=int, default=28)
    parser.add_argument('-num_candidates', type=int, default=10)
    parser.add_argument('-epochs', type=int, default=150)
    parser.add_argument('-threshold', type=int, default=1)
    parser.add_argument('-window_size', type=int, default=9)
    parser.add_argument('-thre', type=int, default=5)
    parser.add_argument('-start', type=int, default=8)
    parser.add_argument('-window', type=int, default=8)
    parser.add_argument('-drop_out', type=float, default=0.4)
    parser.add_argument('-batch_size', type=int, default=32768)

    opt = parser.parse_args()

    dataset_list = ['HDFS', 'openStack', 'KUB', 'BGL']
    dataset_set = set(dataset_list)

    if opt.dir not in dataset_set:
        print('Please enter the correct dataset')
    else:
        print('```````````````````````````````````````')
        pre_train(opt.dir, epochs=opt.epochs, num_classes=opt.num_classes, window_size=opt.window_size,
                  batch_size=opt.batch_size)
        print('```````````````````````````````````````')
        running(opt.start, opt.window, opt.dir, opt.num_classes, opt.num_candidates, opt.threshold, opt.thre,
                opt.window_size,
                opt.drop_out, opt.epochs, opt.batch_size)
        print('End~')
