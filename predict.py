import argparse
import logging
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                image_size=(640, 480),
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, image_size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    # parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
    #                     help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--channels', '-ch', dest='channels', metavar='B', type=int, default=3, help='channels')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def get_all_files_in_folder(folder, types, simple_sort=False):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(Path(folder).rglob(t))
    if simple_sort:
        files_grabbed = sorted(files_grabbed)
    else:
        files_grabbed = sorted(files_grabbed, key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    return files_grabbed

def recreate_folder(path):
    output_dir = Path(path)
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    test_data_path = "data/test_imgs/"
    train_data_path = "data/imgs"
    model_path = "checkpoints/checkpoint_epoch25_npy.pth"
    image_size = (640, 480)

    # in_files = [test_data_path + i for i in os.listdir(test_data_path)]

    in_files = get_all_files_in_folder(test_data_path, ["*"])

    # in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    train_imgs = set(os.listdir(train_data_path))
    test_imgs = set(os.listdir(test_data_path))
    is_contains = len(train_imgs.union(test_imgs)) == 0
    print("Train set contains test? ", is_contains)

    output_masks_dir = 'data/test_masks'
    recreate_folder(output_masks_dir)

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        if filename.name.split('.')[1] == 'npy':
            numpy_image = np.load(filename, allow_pickle=True)
            img = Image.fromarray(np.uint8(numpy_image))
            out_filename = os.path.join(output_masks_dir, f"{filename.stem}.jpg")
        else:
            img = Image.open(filename)
            out_filename = os.path.join(output_masks_dir, filename.name)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           image_size=image_size,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            # out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)

            fig, axs = plt.subplots(ncols=2)
            axs[0].imshow(img)
            axs[1].imshow(result)
            plt.show()
            fig.savefig(out_filename, dpi=fig.dpi)

            # result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
