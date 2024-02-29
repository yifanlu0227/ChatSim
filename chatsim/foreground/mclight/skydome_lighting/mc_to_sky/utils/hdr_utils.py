import numpy as np
from envmap import EnvironmentMap, rotation_matrix
import cv2
import torch

def srgb_gamma_correction(linear_image):
    """
    linear_image: np.ndarray
        shape: H*W*C
    """
    linear_image = np.clip(linear_image, 0, 1)  # 将值限制在0到1之间
    gamma_corrected_image = np.where(linear_image <= 0.0031308,
                          linear_image * 12.92,
                          1.055 * (linear_image ** (1 / 2.4)) - 0.055)
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)  # 将值再次限制在0到1之间

    return gamma_corrected_image

def srgb_inv_gamma_correction(gamma_corrected_image):
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)  # 将值限制在0到1之间
    linear_image = np.where(gamma_corrected_image <= 0.04045,
                            gamma_corrected_image / 12.92,
                            ((gamma_corrected_image + 0.055) / 1.055)**2.4)
    linear_image = np.clip(linear_image, 0, 1)  # 将值再次限制在0到1之间

    return linear_image


def srgb_gamma_correction_torch(linear_image):
    """
    linear_image: torch.tensor
        shape: H*W*C
    """
    linear_image = torch.clamp(linear_image, 0, 1)  # 将值限制在0到1之间
    gamma_corrected_image = torch.where(linear_image <= 0.0031308,
                          linear_image * 12.92,
                          1.055 * (linear_image ** (1 / 2.4)) - 0.055)
    gamma_corrected_image = torch.clamp(gamma_corrected_image, 0, 1)  # 将值再次限制在0到1之间

    return gamma_corrected_image

def srgb_inv_gamma_correction_torch(gamma_corrected_image):
    gamma_corrected_image = torch.clamp(gamma_corrected_image, 0, 1)  # 将值限制在0到1之间
    linear_image = torch.where(gamma_corrected_image <= 0.04045,
                            gamma_corrected_image / 12.92,
                            ((gamma_corrected_image + 0.055) / 1.055)**2.4)
    linear_image = torch.clamp(linear_image, 0, 1)  # 将值再次限制在0到1之间

    return linear_image

def adjust_exposure(image, range=(-2.5,0.5)):
    exposure = np.random.rand() * (range[1] - range[0]) + range[0]
    return image * (2**exposure)

def adjust_flip(image, probability=0.5):
    if np.random.random() < probability:
        return np.fliplr(image)
    else:
        return image
    
def adjust_rotation(image, azimuth=None):
    # azimuth is positive, rotate in right direction
    # azimuth in radians
    # random rotate, not center align the sun.
    if azimuth is None:
        azimuth = np.random.rand()*2*np.pi

    envmap = EnvironmentMap(image, 'skylatlong')
    envmap.rotate(dcm = rotation_matrix(azimuth=azimuth, elevation=0))
    return envmap.data

def adjust_color_temperature(img, temp_range):
    """
    调整图像的色温
    :param img: np.ndarray, 图像数据
    :param temperature: float, 色温调整的比例，大于1表示变暖，小于1表示变冷
    :return: np.ndarray, 色温调整后的图像
    """
    temperature = np.random.rand() * (temp_range[1] - temp_range[0]) + temp_range[0]

    # 调整蓝色通道
    img[:, :, 2] = img[:, :, 2] * temperature
    
    # 调整红色通道
    img[:, :, 0] = img[:, :, 0] / temperature
    
    return img
