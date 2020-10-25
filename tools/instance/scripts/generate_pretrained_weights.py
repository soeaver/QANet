import argparse

import torch

import _init_paths

# Parse arguments
parser = argparse.ArgumentParser(description='QANet')
parser.add_argument('model_path', help='model file (.pth) path', type=str)
parser.add_argument('-s', '--save_path', default='', help='save model path', type=str)
args = parser.parse_args()


def main():
    try:
        weights_dict = torch.load(args.model_path, map_location=torch.device("cpu"))['model']
        save_path = args.save_path if args.save_path else args.model_path.replace('model_', 'weights_')
        torch.save(weights_dict, save_path)
        print('Pre-trained weights is saved to {}'.format(save_path))
    except:
        raise ValueError(args.model_path + ' is not property for generating pre-trained model.')


if __name__ == '__main__':
    main()
