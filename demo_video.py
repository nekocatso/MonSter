import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
# from igev_warp import IGEVStereo_warp
# from igev_refine import IGEVStereo_ref
# from igev_ff import IGEVStereo_ref
from igev_ff_f11 import IGEVStereo_f11, autocast

from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
import torch.nn.functional as F
import sys
import time
# sys.path.append('/data/cjd/DepthAnything_IGEV/Depth-Anything-V2')
# from depth_anything_v2.dpt import DepthAnythingV2

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class NormalizeTensor(object):
    """Normalize a tensor by given mean and std."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            
        Returns:
            Tensor: Normalized Tensor image.
        """
        # Ensure mean and std have the same number of channels as the input tensor
        Device = tensor.device
        self.mean = self.mean.to(Device)
        self.std = self.std.to(Device)

        # Normalize the tensor
        if self.mean.ndimension() == 1:
            self.mean = self.mean[:, None, None]
        if self.std.ndimension() == 1:
            self.std = self.std[:, None, None]

        return (tensor - self.mean) / self.std

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(IGEVStereo_f11(args), device_ids=[0])
    # model.load_state_dict(torch.load(args.restore_ckpt))

    assert os.path.exists(args.restore_ckpt)
    checkpoint = torch.load(args.restore_ckpt)
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # for key in checkpoint:
        # ckpt[key.replace('module.', '')] = checkpoint[key]
        # ckpt['module.' + key] = checkpoint[key]

    model.load_state_dict(checkpoint, strict=True)

    model = model.module
    model.to(DEVICE)
    model.eval()

    # mono_model_configs = {
    #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    # }


    # depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
    # depth_anything.load_state_dict(torch.load(f'/data/cjd/DepthAnything_IGEV/Depth-Anything-V2/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    # depth_anything = depth_anything.cuda().eval()
    # depth_anything.requires_grad_(False)


    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = NormalizeTensor(mean, std)

    videoWrite = cv2.VideoWriter('./ff11_sequence_04.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1242, 750))

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            # DEVICE_img = image1.device
            # image_mono = normalize(image1.clone() / 255.0)
            # height_ori, width_ori = image1.shape[2:]
            # height_mono, width_mono = int(height_ori / 14)*14, int(width_ori / 14)*14
            # image_mono = F.interpolate(image_mono, size=(height_mono, width_mono), mode='bilinear', align_corners=False)
            # depth_mono, features_mono = depth_anything.forward_features(image_mono)
            
            # depth_mono = depth_mono.unsqueeze(0)
            # depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False).to(DEVICE_img).squeeze(0)
            # features_mono = F.interpolate(features_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False).to(DEVICE_img)
            start_time = time.time()
            # disp = model(image1, image2, depth_mono, features_mono, iters=args.valid_iters, test_mode=True)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time:.4f} seconds")
            disp = padder.unpad(disp)
            file_stem = os.path.join(output_directory, imfile1.split('/')[-1]).replace('.png', '')
            disp = disp.cpu().numpy().squeeze()
            # disp = np.round(disp * 256).astype(np.uint16)
            disp_np = (2.0*disp).astype(np.uint8)

            # disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_PLASMA)
            # image_np = np.array(Image.open(imfile1)).astype(np.uint8)       
            # out_img = np.concatenate((image_np, disp_np), 0)
            # cv2.imshow('img', out_img)
            # cv2.waitKey(1)
        #     videoWrite.write(out_img)
        # videoWrite.release()
            # skimage.io.imsave(file_stem, disp)
            # if args.save_numpy:
            np.save(f"{file_stem}.npy", disp_np.squeeze())
            
            # a = input('enter to continue')
            # print(a)
            

            # depth_mono = F.interpolate(depth_mono, size=(int(height_ori/4), int(width_ori/4)), mode='bilinear', align_corners=False)
            # file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            # disp = depth_mono.cpu().numpy().squeeze()
            # disp = (disp / disp.max() * 256)
            # disp = np.round(disp * 256).astype(np.uint16)
            # skimage.io.imsave(file_stem, disp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/home/cjd/cvpr2025/fusion/checkpoint/ff11/mix_k1/100000.pth")
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/data2/cjd/mono_fusion/checkpoints/mix_k2/15000.pth")
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/data2/cjd/mono_fusion/checkpoints/mix_k3/10000.pth")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/home/cjd/cvpr2025/00130000_stereo.pth")
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/home/cjd/cvpr2025/370000_igev_ff5.pth")



    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/data2/cjd/mono_fusion/checkpoints/only_kitti/final.pth")

    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/cjd/stereo_dataset/kitti/2015/testing/image_2/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/cjd/stereo_dataset/kitti/2015/testing/image_3/*_10.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/cjd/stereo_dataset/kitti/2012/testing/colored_0/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/cjd/stereo_dataset/kitti/2012/testing/colored_1/*_10.png")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/cjd/kitti_odometry/dataset/sequences/00/image_2/*.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/cjd/kitti_odometry/dataset/sequences/00/image_3/*.png")

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/cjd/tartanair_true/neighborhood/neighborhood/Hard/P005/image_left/*.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/cjd/tartanair_true/neighborhood/neighborhood/Hard/P005/image_right/*.png")

    parser.add_argument('--output_directory', help="directory to save output", default="neighborhood_Hard_P005")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)