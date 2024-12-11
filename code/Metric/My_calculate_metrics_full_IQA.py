import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
import lpips
import argparse
from basicsr.metrics import calculate_psnr, calculate_ssim
from natsort import natsorted  # 导入natsorted函数

def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_gt', type=str, default='/data/55d_1/IRR-DiffSR/data_samples/UFO-120/TEST/hr')
    parser.add_argument('--folder_restored', type=str, default='/data/55d/results/PDM-SR-origin/test_PDM-SR_UFO120_x4/UFO120_PDM-SR_test_x4')
    parser.add_argument('--output_file', type=str, default='/data/55d/results/PDM-SR-origin/test_PDM-SR_UFO120_x4/results.txt') # the output file name
    args = parser.parse_args()

    psnr_all = []
    ssim_all = []
    lpips_all = []

    # 使用natsorted进行自然排序
    img_list = natsorted(glob.glob(osp.join(args.folder_gt, '*.jpg')))
    lr_list = natsorted(glob.glob(osp.join(args.folder_restored, '*.png')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # initialize the lpips model
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda(0)

    # create a file object
    f = open(args.output_file, 'w')

    for i, (img_path, lr_path) in enumerate(zip(img_list, lr_list)):
        basename_GT, ext_GT = osp.splitext(osp.basename(img_path))
        basename_restored, ext_restored = osp.splitext(osp.basename(lr_path))
        
        # read and process images
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_restored = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        
        # calculate psnr
        print(f'calculating image {i}: {basename_GT},{basename_restored} {img_restored.shape} {img_gt.shape}')
        psnr = calculate_psnr(img_restored, img_gt, crop_border=4, test_y_channel=False)
        psnr_all.append(psnr)

        # calculate ssim
        ssim = calculate_ssim(img_restored, img_gt, crop_border=4, test_y_channel=False)
        ssim_all.append(ssim)

        # calculate lpips
        img_gt_lpips = img_gt.astype(np.float32) / 255.
        img_restored_lpips = img_restored.astype(np.float32) / 255.
        img_gt_lpips, img_restored_lpips = img2tensor([img_gt_lpips, img_restored_lpips], bgr2rgb=True, float32=True)

        # norm to [-1, 1]
        normalize(img_gt_lpips, mean, std, inplace=True)
        normalize(img_restored_lpips, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored_lpips.unsqueeze(0).cuda(0), img_gt_lpips.unsqueeze(0).cuda(0)).cpu().data.numpy()[0, 0, 0, 0]
        lpips_all.append(lpips_val)

        # write the results for each image
        f.write(f'{basename_GT+ext_GT} {basename_restored+ext_restored} PSNR: {psnr:.6f} SSIM: {ssim:.6f} LPIPS: {lpips_val:.6f}\n')

    # calculate the average results
    avg_psnr = sum(psnr_all) / len(psnr_all)
    avg_ssim = sum(ssim_all) / len(ssim_all)
    avg_lpips = sum(lpips_all) / len(lpips_all)

    # write the average results
    f.write(f'Average: PSNR: {avg_psnr:.6f} SSIM: {avg_ssim:.6f} LPIPS: {avg_lpips:.6f}\n')

    # close the file object
    f.close()

    # print the average results
    print(f'Average: PSNR: {avg_psnr:.6f}')
    print(f'Average: SSIM: {avg_ssim:.6f}')
    print(f'Average: LPIPS: {avg_lpips:.6f}')


if __name__ == '__main__':
    main()
