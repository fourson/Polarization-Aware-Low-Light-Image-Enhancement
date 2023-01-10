import os
import argparse
import importlib
import sys

import torch
from tqdm import tqdm
import numpy as np
import cv2


def infer_default(verbose_output=False):
    S0_out_dir = os.path.join(result_dir, 'S0_out')
    S1_out_dir = os.path.join(result_dir, 'S1_out')
    S2_out_dir = os.path.join(result_dir, 'S2_out')
    util.ensure_dir(S0_out_dir)
    util.ensure_dir(S1_out_dir)
    util.ensure_dir(S2_out_dir)

    if verbose_output:
        I1_out_dir = os.path.join(result_dir, 'I1_out')
        I2_out_dir = os.path.join(result_dir, 'I2_out')
        I3_out_dir = os.path.join(result_dir, 'I3_out')
        I4_out_dir = os.path.join(result_dir, 'I4_out')
        AoP_out_dir = os.path.join(result_dir, 'AoP_out')
        DoP_out_dir = os.path.join(result_dir, 'DoP_out')
        util.ensure_dir(I1_out_dir)
        util.ensure_dir(I2_out_dir)
        util.ensure_dir(I3_out_dir)
        util.ensure_dir(I4_out_dir)
        util.ensure_dir(AoP_out_dir)
        util.ensure_dir(DoP_out_dir)

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            name = sample['name'][0]

            # get data and send them to GPU
            # (1, 1, 1) GPU tensor
            amp = sample['amp'].to(device)

            name = name + '_' + str(int(sample['amp'])) + 'x'

            # (N, 3, H, W) GPU tensor
            I1_in = sample['L1'].to(device) * amp
            I2_in = sample['L2'].to(device) * amp
            I3_in = sample['L3'].to(device) * amp
            I4_in = sample['L4'].to(device) * amp
            S0_in, S1_in, S2_in, DoP_in, AoP_in = util.compute_Si_from_Ii(I1_in, I2_in, I3_in, I4_in)
            # normalize to [0, 1]
            S0_in = S0_in / 2
            S1_in = (S1_in + 1) / 2
            S2_in = (S2_in + 1) / 2

            # get network output
            # (N, 3, H, W) GPU tensor
            S0_out, S1_out, S2_out = model(S0_in, S1_in, S2_in)

            # denormalize
            S0_out = S0_out * 2
            S1_out = S1_out * 2 - 1
            S2_out = S2_out * 2 - 1

            # save data
            S0_out_numpy = np.transpose(S0_out.squeeze().cpu().numpy(), (1, 2, 0))
            S1_out_numpy = np.transpose(S1_out.squeeze().cpu().numpy(), (1, 2, 0))
            S2_out_numpy = np.transpose(S2_out.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(S0_out_dir, name + '.npy'), S0_out_numpy)
            np.save(os.path.join(S1_out_dir, name + '.npy'), S1_out_numpy)
            np.save(os.path.join(S2_out_dir, name + '.npy'), S2_out_numpy)

            if verbose_output:
                I1_out, I2_out, I3_out, I4_out, DoP_out, AoP_out = util.compute_Ii_from_Si(S0_out, S1_out, S2_out)

                I1_out_numpy = np.transpose(I1_out.squeeze().cpu().numpy(), (1, 2, 0))
                I2_out_numpy = np.transpose(I2_out.squeeze().cpu().numpy(), (1, 2, 0))
                I3_out_numpy = np.transpose(I3_out.squeeze().cpu().numpy(), (1, 2, 0))
                I4_out_numpy = np.transpose(I4_out.squeeze().cpu().numpy(), (1, 2, 0))
                AoP_out_numpy = np.transpose(AoP_out.squeeze().cpu().numpy(), (1, 2, 0))
                DoP_out_numpy = np.transpose(DoP_out.squeeze().cpu().numpy(), (1, 2, 0))
                cv2.imwrite(os.path.join(I1_out_dir, name + '.png'),
                            cv2.cvtColor(I1_out_numpy, cv2.COLOR_RGB2BGR) * 255)
                cv2.imwrite(os.path.join(I2_out_dir, name + '.png'),
                            cv2.cvtColor(I2_out_numpy, cv2.COLOR_RGB2BGR) * 255)
                cv2.imwrite(os.path.join(I3_out_dir, name + '.png'),
                            cv2.cvtColor(I3_out_numpy, cv2.COLOR_RGB2BGR) * 255)
                cv2.imwrite(os.path.join(I4_out_dir, name + '.png'),
                            cv2.cvtColor(I4_out_numpy, cv2.COLOR_RGB2BGR) * 255)
                np.save(os.path.join(AoP_out_dir, name + '.npy'), AoP_out_numpy)
                np.save(os.path.join(DoP_out_dir, name + '.npy'), DoP_out_numpy)


def infer_gray(verbose_output=False):
    S0_out_dir = os.path.join(result_dir, 'S0_out')
    S1_out_dir = os.path.join(result_dir, 'S1_out')
    S2_out_dir = os.path.join(result_dir, 'S2_out')
    util.ensure_dir(S0_out_dir)
    util.ensure_dir(S1_out_dir)
    util.ensure_dir(S2_out_dir)

    if verbose_output:
        I1_out_dir = os.path.join(result_dir, 'I1_out')
        I2_out_dir = os.path.join(result_dir, 'I2_out')
        I3_out_dir = os.path.join(result_dir, 'I3_out')
        I4_out_dir = os.path.join(result_dir, 'I4_out')
        AoP_out_dir = os.path.join(result_dir, 'AoP_out')
        DoP_out_dir = os.path.join(result_dir, 'DoP_out')
        util.ensure_dir(I1_out_dir)
        util.ensure_dir(I2_out_dir)
        util.ensure_dir(I3_out_dir)
        util.ensure_dir(I4_out_dir)
        util.ensure_dir(AoP_out_dir)
        util.ensure_dir(DoP_out_dir)

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            name = sample['name'][0]

            # get data and send them to GPU
            # (1, 1, 1) GPU tensor
            amp = sample['amp'].to(device)

            name = name + '_' + str(int(sample['amp'])) + 'x'

            # (N, 3, H, W) GPU tensor
            I1_in = sample['L1'].to(device) * amp
            I2_in = sample['L2'].to(device) * amp
            I3_in = sample['L3'].to(device) * amp
            I4_in = sample['L4'].to(device) * amp
            S0_in, S1_in, S2_in, DoP_in, AoP_in = util.compute_Si_from_Ii(I1_in, I2_in, I3_in, I4_in)
            # normalize to [0, 1]
            S0_in = S0_in / 2
            S1_in = (S1_in + 1) / 2
            S2_in = (S2_in + 1) / 2

            # get network output
            # (N, 3, H, W) GPU tensor
            S0_out, S1_out, S2_out = model(S0_in, S1_in, S2_in)

            # denormalize
            S0_out = S0_out * 2
            S1_out = S1_out * 2 - 1
            S2_out = S2_out * 2 - 1

            # save data
            S0_out_numpy = S0_out.squeeze().cpu().numpy()
            S1_out_numpy = S1_out.squeeze().cpu().numpy()
            S2_out_numpy = S2_out.squeeze().cpu().numpy()
            np.save(os.path.join(S0_out_dir, name + '.npy'), S0_out_numpy)
            np.save(os.path.join(S1_out_dir, name + '.npy'), S1_out_numpy)
            np.save(os.path.join(S2_out_dir, name + '.npy'), S2_out_numpy)

            if verbose_output:
                I1_out, I2_out, I3_out, I4_out, DoP_out, AoP_out = util.compute_Ii_from_Si(S0_out, S1_out, S2_out)

                I1_out_numpy = I1_out.squeeze().cpu().numpy()
                I2_out_numpy = I2_out.squeeze().cpu().numpy()
                I3_out_numpy = I3_out.squeeze().cpu().numpy()
                I4_out_numpy = I4_out.squeeze().cpu().numpy()
                AoP_out_numpy = AoP_out.squeeze().cpu().numpy()
                DoP_out_numpy = DoP_out.squeeze().cpu().numpy()
                cv2.imwrite(os.path.join(I1_out_dir, name + '.png'), I1_out_numpy * 255)
                cv2.imwrite(os.path.join(I2_out_dir, name + '.png'), I2_out_numpy * 255)
                cv2.imwrite(os.path.join(I3_out_dir, name + '.png'), I3_out_numpy * 255)
                cv2.imwrite(os.path.join(I4_out_dir, name + '.png'), I4_out_numpy * 255)
                np.save(os.path.join(AoP_out_dir, name + '.npy'), AoP_out_numpy)
                np.save(os.path.join(DoP_out_dir, name + '.npy'), DoP_out_numpy)


if __name__ == '__main__':
    MODULE = 'subnetwork2'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--data_dir', required=True, type=str, help='dir of input data')
    parser.add_argument('--result_dir', required=True, type=str, help='dir to save result')
    parser.add_argument('--data_loader_type', default='InferDataLoader', type=str, help='which data loader to use')
    parser.add_argument('--verbose_output', default=0, type=int, help='output I_{1,2,3,4}, AoP, and DoP')
    subparsers = parser.add_subparsers(help='which func to run', dest='func')

    # add subparsers and their args for each func
    subparser_default = subparsers.add_parser("default")
    subparser_gray = subparsers.add_parser("gray")

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PATH
    from utils import util

    # load checkpoint
    checkpoint = torch.load(args.resume)
    config = checkpoint['config']
    assert config['module'] == MODULE

    # setup data_loader instances
    # we choose batch_size=1(default value)
    # module_data = importlib.import_module('.data_loader_' + MODULE, package='data_loader')
    module_data = importlib.import_module('.data_loader', package='data_loader')  # share the same dataloader
    data_loader_class = getattr(module_data, args.data_loader_type)
    data_loader = data_loader_class(data_dir=args.data_dir)

    # build model architecture
    module_arch = importlib.import_module('.model_' + MODULE, package='model')
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # prepare model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # set the model to validation mode
    model.eval()

    # ensure result_dir
    result_dir = args.result_dir
    util.ensure_dir(result_dir)

    # run the selected func
    if args.func == 'default':
        infer_default(args.verbose_output)
    elif args.func == 'gray':
        # run the default
        infer_gray(args.verbose_output)
    else:
        infer_default(args.verbose_output)
