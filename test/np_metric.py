import torch as t
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import tensorflow as tf

def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return t.mean(t.abs(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))


def cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return t.mean(t.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))


def cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return t.sqrt(t.mean(t.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))


def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    return t.mean(t.sqrt(t.mean(t.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]), axis=[2, 3])))


def cloud_ssim(y_true, y_pred):
    """Computes the SSIM over the full image."""
    y_true = y_true[:, 0:13, :, :].clone()
    y_pred = y_pred[:, 0:13, :, :].clone()

    y_true *= 2000
    y_pred *= 2000
    y_true = y_true.permute(0, 2, 3, 1)
    y_pred = y_pred.permute(0, 2, 3, 1)
    y_true=np.asarray(y_true.cpu())
    y_pred=np.asarray(y_pred.cpu())
    
    # y_true = tf.transpose(y_true, [0, 2, 3, 1])
    # y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    ssim = tf.image.ssim(y_true, y_pred, max_val=10000.0)
    ssim = tf.reduce_mean(ssim)


    #ssim = compare_ssim(y_true, y_pred, multichannel=True, win_size=51)
    
    return ssim


def cloud_psnr(y_true, y_pred):
    """Computes the PSNR over the full image."""
    y_true = y_true[:, 0:13, :, :].clone()
    y_pred = y_pred[:, 0:13, :, :].clone()
    y_true *= 2000
    y_pred *= 2000
    #s2img = (np.clip(s2img, 0, 10000) / self.scale).astype('float32')
    #y_pred = np.clip(y_pred, 0, 10000)
    
    rmse = t.sqrt(t.mean(t.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))

    return 20.0 * (t.log(10000.0 / rmse) / t.log(t.tensor(10.0)))
    #return 10 * np.log10((10000.0 ** 2) / rmse.cpu())

    # y_true = np.asarray(y_true.cpu())
    # y_pred = np.asarray(y_pred.cpu())
    # 
    # p1 = compare_psnr(y_true, y_pred,image_true=False)
    #return p1


