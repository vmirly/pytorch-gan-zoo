import sys
import argparse
import torch


def main(args):
    print(args)
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.ngpu > 0)
        else "cpu")
    print('Device:', device)

def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', default=0, help='Number of GPUs to use')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse(sys.argv[1:])
    main(args)
