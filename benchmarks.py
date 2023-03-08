import argparse


def HDFS():
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


def openStack():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default='openStack')
    parser.add_argument('-num_classes', type=int, default=1145)
    parser.add_argument('-num_candidates', type=int, default=433)
    parser.add_argument('-epochs', type=int, default=150)
    parser.add_argument('-threshold', type=int, default=14)
    parser.add_argument('-window_size', type=int, default=50)
    parser.add_argument('-thre', type=int, default=25)
    parser.add_argument('-start', type=int, default=610)
    parser.add_argument('-window', type=int, default=10)
    parser.add_argument('-drop_out', type=float, default=0.4)
    parser.add_argument('-batch_size', type=int, default=4096)

def BGL():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default='BGL')
    parser.add_argument('-num_classes', type=int, default=433)
    parser.add_argument('-num_candidates', type=int, default=15)
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-threshold', type=int, default=2)
    parser.add_argument('-window_size', type=int, default=5)
    parser.add_argument('-thre', type=int, default=3)
    parser.add_argument('-start', type=int, default=12)
    parser.add_argument('-window', type=int, default=6)
    parser.add_argument('-drop_out', type=float, default=0.6)
    parser.add_argument('-batch_size', type=int, default=32768)

def KUB():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default='KUB')
    parser.add_argument('-num_classes', type=int, default=591)
    parser.add_argument('-num_candidates', type=int, default=10)
    parser.add_argument('-epochs', type=int, default=150)
    parser.add_argument('-threshold', type=int, default=2)
    parser.add_argument('-window_size', type=int, default=10)
    parser.add_argument('-thre', type=int, default=5)
    parser.add_argument('-start', type=int, default=8)
    parser.add_argument('-window', type=int, default=6)
    parser.add_argument('-drop_out', type=float, default=0.2)
    parser.add_argument('-batch_size', type=int, default=32768)