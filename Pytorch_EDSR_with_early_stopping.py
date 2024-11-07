import os, time, scipy.io
import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

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
parser.add_argument('--scale', type=str, default=2,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size')
parser.add_argument('--n_colors', type=int, default=4,
                    help='number of input color channels to use')
parser.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args = parser.parse_args()



home_dir = '/data/yaishof/NN_project/'

input_dir = home_dir+'Sony/short/'
gt_dir = home_dir+'Sony/long/'
result_dir = home_dir+'DeepImageDenoising/result_Sony/'
model_dir = home_dir+'DeepImageDenoising//saved_model_Sony/'
test_name = 'ES_edsr-lerelu-ps-'+str(args.patch_size)+'-b-'+str(args.n_resblocks)+'/'


# get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i]) #take the file name only
    train_ids.append(int(train_fn[0:5]))

# get validation IDs
validation_fns = glob.glob(gt_dir + '/2*.ARW')
validation_ids = []
for i in range(len(validation_fns)):
    _, validation_fn = os.path.split(validation_fns[i])
    validation_ids.append(int(validation_fn[0:5]))
validation_gt_images=[None]*6000
validation_input_images = {}
validation_input_images['300'] = [None]*len(validation_ids)
validation_input_images['250'] = [None]*len(validation_ids)
validation_input_images['100'] = [None]*len(validation_ids)



ps = args.patch_size  # patch size for training
save_freq = 25

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


# Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['300'] = [None]*len(train_ids)
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
model = EDSR(args)


######################################################################
# m_path = home_dir+'DeepImageDenoising//saved_model_Sony/'+'edsr-lerelu-ps-'+str(args.patch_size)+'-b-'+str(args.n_resblocks)+'/'
# m_name = 'edsr_sony_e1325.pth'
# model.load_state_dict(torch.load(m_path + m_name))
# lastepoch = 1326
#####################################################################


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.cuda()


avg_validation_loss_list = [np.inf]
epochs_no_improve = 0 
N_EPOCHS_STOP = 25

best_epoch = 0
min_validation_loss = np.inf

opt = optim.Adam(model.parameters(), lr=learning_rate)
EPOCH_NUM = 6001
for epoch in range(lastepoch, EPOCH_NUM):
    if os.path.isdir("result/%04d" % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        for g in opt.param_groups:
            g['lr'] = 1e-5

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.randint(0, len(in_files))]
        _, in_fn = os.path.split(in_path) #for example in_fn='00047_00_0.1s.ARW'

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0] 
        _, gt_fn = os.path.split(gt_path) #for example gt_fn='00047_00_10s.ARW'
        in_exposure = float(in_fn[9:-5]) # '0.1' in the the example 
        gt_exposure = float(gt_fn[9:-5]) # '10' in the the example 
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        in_img = torch.from_numpy(input_patch).permute(0, 3, 1, 2).cuda()
        gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2).cuda()

        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        opt.step()
        g_loss[ind] = loss.cpu().data # was loss.data

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))
    ##############################################################################
    #check results on validation set and preform early stopping if needed
    #torch.cuda.empty_cache()
    with torch.no_grad():
        avg_loss_validation = 0
        validation_sample_number = 0
        for i in range(len(validation_ids)):
            validation_id = validation_ids[i]
            # test the first image in each sequence
            in_files = glob.glob(input_dir + '%05d_00*.ARW' % validation_id)
            for k in range(len(in_files)):
                in_path = in_files[k]
                _, in_fn = os.path.split(in_path)
                #print(in_fn)
                gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % validation_id)
                gt_path = gt_files[0]
                _, gt_fn = os.path.split(gt_path)
                in_exposure = float(in_fn[9:-5])
                gt_exposure = float(gt_fn[9:-5])
                ratio = min(gt_exposure / in_exposure, 300)
                if validation_input_images[str(ratio)[0:3]][i] is None:
                    raw = rawpy.imread(in_path)
                    validation_input_images[str(ratio)[0:3]][i] = np.expand_dims(pack_raw(raw), axis=0) * ratio
                    if validation_gt_images[i] is None:
                        gt_raw = rawpy.imread(gt_path)
                        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                        validation_gt_images[i] = np.expand_dims(np.float32(im / 65535.0), axis=0)

                # crop
                H = validation_input_images[str(ratio)[0:3]][i].shape[1]
                W = validation_input_images[str(ratio)[0:3]][i].shape[2]

                xx = np.random.randint(0, W - ps)
                yy = np.random.randint(0, H - ps)
                input_patch = validation_input_images[str(ratio)[0:3]][i][:, yy:yy + ps, xx:xx + ps, :]
                gt_patch = validation_gt_images[i][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

                input_patch = np.minimum(input_patch, 1.0)
                gt_patch = np.maximum(gt_patch, 0.0)

                in_img = torch.from_numpy(input_patch).permute(0, 3, 1, 2).cuda()
                gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2).cuda()
                out_img = model(in_img)
                validation_loss = reduce_mean(out_img, gt_img)
                avg_loss_validation += validation_loss
                validation_sample_number += 1
        avg_loss_validation /= validation_sample_number
        avg_validation_loss_list.append(avg_loss_validation.cpu().data.item())
        if (avg_validation_loss_list[-1]>=min_validation_loss):
            epochs_no_improve += 1
        else: 
            epochs_no_improve = 0
            min_validation_loss = avg_validation_loss_list[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), model_dir + test_name + 'best_model.pth')
            

    if (epoch % save_freq == 0 or epochs_no_improve >= N_EPOCHS_STOP):
        if not os.path.isdir(model_dir + test_name):
            os.makedirs(model_dir + test_name)
            
        torch.save(model.state_dict(), model_dir + test_name + 'edsr_sony_e%04d.pth' % epoch)
        if (epochs_no_improve == N_EPOCHS_STOP):
            break




print(avg_validation_loss_list)
print("best_epoch:", best_epoch)
print("min_validation_loss:", min_validation_loss)