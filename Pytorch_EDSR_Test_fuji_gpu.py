'''
########## 1024x1024 ##########

32 block EDSR with SE:
mean psnr: 29.57
mean ssim: 0.8889

16 block EDSR with SE:
mean psnr: 29.42
mean ssim: 0.8865

UNET Resulte:
mean psnr:  29.52
mean ssim:   0.8897

############ Full ############
32 block EDSR with SE:
mean psnr: 29.57
mean ssim: 0.8889

16 block EDSR with SE:
mean psnr: 29.42
mean ssim: 0.8865

UNET Resulte:
mean psnr:  29.52
mean ssim:   0.8897

'''

import os, time, scipy.io

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
import skimage.metrics as skm #changed from skimage.measure
import cv2
from skimage import exposure
from PIL import Image


from EDSR import EDSR
import argparse




# Argument for EDSR
parser = argparse.ArgumentParser(description='EDSR')
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--scale', type=str, default=3,
                    help='super resolution scale')
parser.add_argument('--scale_type', type=str, default='upConv',
                    help='super resolution scale type')                    
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size')
parser.add_argument('--n_colors', type=int, default=9,
                    help='number of input color channels to use')
parser.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args = parser.parse_args()

print(args.scale_type)

torch.manual_seed(0)
home_dir = '/data/yaishof/NN_project/'

input_dir = home_dir+'Fuji/short/'
gt_dir = home_dir+'Fuji/long/'

if (args.scale_type == 'upConv'):
    m_path = home_dir+'DeepImageDenoising/saved_model_Fuji/new_with_ConvTranspose2d_edsr-lerelu-ps-256-b-32/'
    m_name = 'edsr_fuji_e4000.pth'
    result_dir = home_dir+'DeepImageDenoising/Final/test_new_with_ConvTranspose2d_edsr_lerelu_ps_256_b_32_edsr_fuji_e4000/'
elif (args.scale_type == 'upNeighbor'):
    m_path = home_dir+'DeepImageDenoising/saved_model_Fuji/new_edsr-lerelu-ps-256-b-32/'
    m_name = 'edsr_fuji_e4000.pth'
    result_dir = home_dir+'DeepImageDenoising/Final/upNeighbor_edsr_lerelu_ps_256_b_32_edsr_fuji_e4000/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.RAF')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

ps = args.patch_size  # patch size for training

def pack_raw(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # 1 B
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # 4 R
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # 5 B
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]
    return out


model = EDSR(args)
model.load_state_dict(torch.load(m_path + m_name))
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.cuda()

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

psnr = []
ssim = []
cnt = 0
with torch.no_grad():
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00*.RAF' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.RAF' % test_id)
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
            # input_full = input_full[:,:512, :512, :]

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # im = im[:1024,:1024]
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
            # scale_full = np.minimum(scale_full, 1.0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # im = im[:1024, :1024]
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            input_full = np.minimum(input_full, 1.0)

            in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).type(torch.FloatTensor)#.cuda()
            B, C, H, W = in_img.shape
            output = np.zeros((B, args.scale*H, args.scale*W, args.o_colors))
            num_of_300_blocks = int(H // 300)
            block_sizes_list = [300 for i in range(num_of_300_blocks)]+[H-300*num_of_300_blocks]
            print(block_sizes_list)
            current_index_input = 0
            current_index_output =0 
            
            total_time = 0
            cnt +=1
            for i in range(len(block_sizes_list)):
                next_index_input = current_index_input + block_sizes_list[i]
                input_gpu  = in_img[:, :,current_index_input:next_index_input,:].cuda()
                st = time.time()
                out_img = model(input_gpu)
                total_time += time.time()-st
                current_index_input = next_index_input

                next_index_output = current_index_output + args.scale*block_sizes_list[i]
                output[:, current_index_output:next_index_output, :, :] = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
                current_index_output = next_index_output
            print('%d\tTime without memory loading: %.3f'%(cnt, total_time))

            output = np.minimum(np.maximum(output, 0), 1)
            output = output.astype('float32')

            _, H, W, _ = output.shape
            output = output[0, :, :, :]
            gt_full = gt_full[0, :H, :W, :]
            scale_full = scale_full[0, :H, :W, :]
            origin_full = scale_full
            scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)  # scale the low-light image to the same mean of the groundtruth

            psnr.append(skm.peak_signal_noise_ratio(gt_full[:, :, :], output[:, :, :]))
            ssim.append(skm.structural_similarity(gt_full[:, :, :], output[:, :, :], multichannel=True))
            print('psnr: ', psnr[-1], 'ssim: ', ssim[-1])
            # temp = np.concatenate((scale_full_, gt_full_, output_), axis=1)
            # plt.clf()
            # plt.imshow(temp)
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)
            Image.fromarray((origin_full * 255).astype(np.uint8)).save(
                result_dir + '%5d_00_%d_ori.png' % (test_id, ratio))
            Image.fromarray((output * 255).astype(np.uint8)).save(
                result_dir + '%5d_00_%d_out.png' % (test_id, ratio))
            Image.fromarray((scale_full * 255).astype(np.uint8)).save(
                result_dir + '%5d_00_%d_scale.png' % (test_id, ratio))
            Image.fromarray((gt_full * 255).astype(np.uint8)).save(
                result_dir + '%5d_00_%d_gt.png' % (test_id, ratio))

print('mean psnr: ', np.mean(psnr))
print('mean ssim: ', np.mean(ssim))
print('done')
