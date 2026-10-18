import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np
from sewar.full_ref import mse, rmse, psnr, ssim, sam
import np_metric as img_met
import metrics_glf_cr as metrics_glf_cr
from torchvision.transforms import Resize
import logging
import datetime
import sys
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.models.archs.dictcr import Restormer
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import matplotlib.pyplot as plt
import numpy as np

import os

import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(level=logging.CRITICAL,
                    format='%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d(%a)%H:%M:%S',
                    filename='test_DSen2_CR_log_' + now + '.txt',
                    filemode='w')

#将大于或等于INFO级别的日志信息输出到StreamHandler(默认为标准错误)
console = logging.StreamHandler()
console.setLevel(logging.CRITICAL)
formatter = logging.Formatter('[%(levelname)-8s] %(message)s')  #屏显实时查看，无需时间
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

parser = argparse.ArgumentParser(description='Test on your own images')
# parser.add_argument('--input_dir', default='/data/wy/code/Mamba/org-restormer/ORG-Restormer-main/Deraining/Datasets/RICE1-inpainting/RICE1-test//cloud/', type=str, help='Directory of input images or path of single image')
# parser.add_argument('--input_truth_dir', default='/data/wy/code/Mamba/org-restormer/ORG-Restormer-main/Deraining/Datasets/RICE1-inpainting/RICE1-test/label/', type=str, help='Directory of input images or path of single image')
# parser.add_argument('--input_dir', default='/home/lanyun/xjtu/hwl/datasets/cloud_wy/RICE1/test/cloud', type=str, help='Directory of input images or path of single image')
# parser.add_argument('--input_truth_dir', default='/home/lanyun/xjtu/hwl/datasets/cloud_wy/RICE1/test/reference', type=str, help='Directory of input images or path of single image')

# parser.add_argument('--input_dir', default='/home/lanyun/xjtu/hwl/datasets/cloud_wy/T-Cloud/test/cloud/', type=str, help='Directory of input images or path of single image')
# parser.add_argument('--input_truth_dir', default='/home/lanyun/xjtu/hwl/datasets/cloud_wy/T-Cloud/test/reference/', type=str, help='Directory of input images or path of single image')

parser.add_argument('--input_dir', default='/data/huang/datasets/cloud/clound-wy/RICE1/test/cloud/', type=str,
                    help='Directory of input images or path of single image')
parser.add_argument('--input_truth_dir', default='/data/huang/datasets/cloud/clound-wy/RICE1/test/reference/',
                    type=str, help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default='./output/rice1_dict_tiny_new1_tokens128_512_1000_train_20241209_512_25000iters_test', type=str,
                    help='Directory for restored results')
parser.add_argument('--task', default='Deraining', type=str, help='Task to run', choices=['Deraining'])
parser.add_argument('--tile', type=int, default=None,
                    help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
# parser.add_argument('--weights', default='/data/wy/code/Mamba/Restormer-main/experiments/Deraining_MambaRestormer_128_multiscal/models/net_g_10000.pth', type=str, help='Path to weights')
args = parser.parse_args()


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)


def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)


def get_weights_and_parameters(task, parameters):
    if task == 'Deraining':
        weights = os.path.join('./experiments/rice1_dict_tiny_new1_tokens128/models/', 'net_g_25000.pth')
    return weights, parameters


task = args.task
inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir)
inp_truth_dir = args.input_truth_dir

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.' + ext)))
    files = natsorted(files)

if any([inp_truth_dir.endswith(ext) for ext in extensions]):
    files_truth = [inp_truth_dir]
else:
    files_truth = []
    for ext in extensions:
        files_truth.extend(glob(os.path.join(inp_truth_dir, '*.' + ext)))
    files_truth = natsorted(files_truth)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters
parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [1, 2, 8,  2, 1], 'heads': [2, 2, 8, 2 ,2], 'num_tokens': [128, 128, 128],
              'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'WithBias'}
weights, parameters = get_weights_and_parameters(task, parameters)

#load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'ShuntRestormer_arch.py'))
#model = load_arch['ShuntRestormer'](**parameters)
# load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'vision_mamba_multi_scal.py'))
model = Restormer(**parameters)  #load_arch['MambaRestormer'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


#test param and flops
# Model Size####
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.3fM" % (total / 1e6))
# print(128, 128, model.flops([128, 128]) / 1e9, 'G')
# print(256, 256, model.flops([256, 256]) / 1e9, 'G')

# Test
_input = torch.randn([1, 3, 256, 256]).cuda()
model = model.cuda() #.cpu()
from fvcore.nn import FlopCountAnalysis, flop_count_table

flops = FlopCountAnalysis(model, inputs=(_input))
print(flop_count_table(flops))

# import time
#
# start_time = time.perf_counter_ns()
# count = 100
# for i in range(count):
#     output = model(_input)
#
# end_time = time.perf_counter_ns()
# ex_time = ((end_time - start_time) / count) * 0.000001
# print(str(ex_time) + "ms")
# print(output.shape)
######
checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

#########################################################
# import yaml

# try:
#     from yaml import CLoader as Loader
# except ImportError:
#     from yaml import Loader
# yaml_file = '/data/wy/code/Mamba/Restormer-main/Deraining/Options/rice2_vison_mamba_128.yml'
# x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

# s = x['network_g'].pop('type')

# model = MambaRestormer(**x['network_g'])

# checkpoint = torch.load(args.weights)
# model.load_state_dict(checkpoint['params'])
# print("===>Testing using weights: ",args.weights)
# model.cuda()
# # model = torch.nn.DataParallel(model)
# model.eval()
# plot sattention map


img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")

MAE_vs = []
MSE_vs = []
RMSE_vs = []
BRMSE_vs = []
ssim_vs = []
psnr_vs = []
sam_vs = []
logging.critical("start testing...")



with torch.no_grad():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if task == 'Deraining':
            img = load_img(file_)
            truth_file = file_.replace(args.input_dir, args.input_truth_dir)
            img_truth = load_img(truth_file)
            # img = cv2.resize(img,(256,256))
            # img_truth = cv2.resize(img_truth, (256, 256))

        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
        height, width = input_.shape[2], input_.shape[3]
        crop_size = 512
        y = np.maximum(0, height - crop_size) // 2
        x = np.maximum(0, width - crop_size) // 2
        input_ = input_[..., y:y + crop_size, x:x + crop_size]

        # torch_resize = Resize([16, 16])
        # input_ = torch_resize(input_)

        input_truth = torch.from_numpy(img_truth).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
        input_truth = input_truth[..., y:y + crop_size, x:x + crop_size]
        # torch_resize = Resize([16, 16])
        # input_truth = torch_resize( input_truth)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        if args.tile is None:
            ## Testing on the original resolution image

            restored = model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :, :height, :width]

        s2img1 = input_truth.clone()  #0-1 value
        fake_img1 = restored.clone()  # convert values from 0-5 to 0-1

        # s2img1 = (s2img1 + 1) / 2.0
        # fake_img1 = (fake_img1 + 1) / 2.0
        MAE_v = img_met.cloud_mean_absolute_error(s2img1, fake_img1)
        MSE_v = img_met.cloud_mean_squared_error(s2img1, fake_img1)
        RMSE_v = img_met.cloud_root_mean_squared_error(s2img1, fake_img1)
        BRMSE_v = img_met.cloud_bandwise_root_mean_squared_error(s2img1, fake_img1)
        # ssim_v = img_met.cloud_ssim(s2img1, fake_img1)
        # psnr_v = img_met.cloud_psnr(s2img1, fake_img1) # equal to glf_cr.psnr
        psnr_v = metrics_glf_cr.PSNR(s2img1, fake_img1)
        ssim_v = metrics_glf_cr.SSIM(s2img1, fake_img1) # torch 1.8.1, 2.1.1 more than torch 2.4.1 0.06. very big gap. please test on torch 1.8.1, 2.1.1
        # ssim_v, _ = ssim(np.array(s2img1.permute(0,2,3,1).cpu()[0]),
        #             np.array(fake_img1.permute(0,2,3,1).cpu()[0]),MAX=1,ws=7)
        # from skimage.metrics import structural_similarity as ssim
        # ssim_v = ssim(np.array(s2img1.permute(0,2,3,1).cpu()[0]),
        #               np.array(fake_img1.permute(0,2,3,1).cpu()[0]),
        #               win_size=11,data_range=1,channel_axis=2,gaussian_weights=True)

        MAE_vs.append(np.asarray(MAE_v.cpu()))
        MSE_vs.append(np.asarray(MSE_v.cpu()))
        RMSE_vs.append(np.asarray(RMSE_v.cpu()))
        BRMSE_vs.append(np.asarray(BRMSE_v.cpu()))
        ssim_vs.append(np.asarray(ssim_v.cpu().detach().numpy()))
        # ssim_vs.append(np.asarray(ssim_v))
        psnr_vs.append(np.asarray(psnr_v))

        #fake_img1 = fake_img1.permute(0, 2, 3, 1)
        #s2img1 = s2img1.permute(0, 2, 3, 1)

        #y_f = np.asarray(fake_img1.cpu().detach().numpy())
        #y_true = np.asarray(s2img1.cpu())

        #sam_v = sam(y_true[0], y_f[0]) * 180 / np.pi  # sam and sam near, only little difference.
        #sam_vs.append(np.asarray(sam_v))

        # spectral angle mapper
        mat = s2img1 * fake_img1
        mat = torch.sum(mat, 1)
        mat = torch.div(mat, torch.sqrt(torch.sum(s2img1 * s2img1, 1)))
        mat = torch.div(mat, torch.sqrt(torch.sum(fake_img1 * fake_img1, 1)))
        sam_v = torch.mean(torch.acos(torch.clamp(mat, -1, 1)) * torch.tensor(180) / np.pi)

        if not torch.isnan(sam_v):
            sam_vs.append(np.asarray(sam_v.cpu().detach().numpy()))



        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        input_1 = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
        input_save = img_as_ubyte(input_1[0])
        input_tr = input_truth.permute(0, 2, 3, 1).cpu().detach().numpy()
        input_tr_save = img_as_ubyte(input_tr[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]
        # stx()
        if task == 'Deraining':
            # save_img((os.path.join(out_dir, f+'.png')), restored)
            save_img((os.path.join(out_dir, f + '_out.png')), restored)
            save_img((os.path.join(out_dir, f + '_input.png')), input_save)
            save_img((os.path.join(out_dir, f + '_truth.png')), input_tr_save)

        # from skimage.metrics import structural_similarity as ssim
        # from PIL import Image
        # img1 = np.array(Image.open(os.path.join(out_dir, f + '_truth.png')))
        # img2 = np.array(Image.open(os.path.join(out_dir, f + '_out.png')))
        # ssim_v = ssim(img1,
        #               img2,
        #               win_size=11, data_range=255, channel_axis=2, gaussian_weights=True)
        # ssim_vs.append(np.asarray(ssim_v)) # less than metrics_glf_cr.SSIM 0.003, different skimage version has the same value.
        #
        # from skimage.metrics import peak_signal_noise_ratio as compare_psnr
        #
        # psnr_v= compare_psnr(img1, img2) # less than metrics_glf_cr.PSNR 0.1.
        #
        # psnr_vs.append(np.asarray(psnr_v))

        logging.critical(
            "MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f},ssim_v:{:.6f},sam_v:{:.6f}, file_name:{}".format(
                MAE_v.cpu(), MSE_v.cpu(),
                RMSE_v.cpu(), BRMSE_v.cpu(),
                psnr_v, ssim_v.cpu(), sam_v, file_))


    MAE_v = np.mean(MAE_vs)
    MSE_v = np.mean(MSE_vs)
    RMSE_v = np.mean(RMSE_vs)
    BRMSE_v = np.mean(BRMSE_vs)
    ssim_v = np.mean(ssim_vs)
    psnr_v = np.mean(psnr_vs)
    sam_v = np.mean(sam_vs)
    # print("MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f}".format(MAE_v, MSE_v, RMSE_v, BRMSE_v,
    #                                                                          psnr_v))
    logging.critical(
        "MAE_m:{:.6f},MSE_m:{:.6f},RMSE_m:{:.6f},BRMSE_m:{:.6f},psnr_m:{:.6f},ssim_m:{:.6f},sam_m:{:.6f}".format(
            MAE_v, MSE_v, RMSE_v, BRMSE_v,
            psnr_v, ssim_v, sam_v))

    print(f"\nRestored images are saved at {out_dir}")
