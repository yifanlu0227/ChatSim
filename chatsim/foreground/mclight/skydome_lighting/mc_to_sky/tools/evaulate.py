# quantitative evaluation
# 1. Peak intensity on HDR dataset
# 2. Peak direction on Holicity dataset
import os
import glob
import imageio.v2 as imageio
import numpy as np
from termcolor import colored
from envmap import EnvironmentMap
from icecream import ic
from mc_to_sky.utils.hdr_utils import srgb_inv_gamma_correction

def evaluate_peak_intensity(visualization_path):
    """
    peak_intensity_error_percentage : float
       (abs(pred_int - gt_int) / gt_int) * 100%

    Args: 
    
        visualization_path : str
            visulization is result on testset. 
            e.g. mc_to_sky/logs/pred_hdr_pano_from_AvgMultiView_enhanced_elu_white_balance_adjust3/lightning_logs/version_0/visualization
            include 'xxxx_hdr_gt.exr', 'xxxx_hdr_pred.exr'
    
    Returns:
        max, min, mean, median of peak_intensity_error_percentage
    """
    hdr_pred_files = sorted(glob.glob(os.path.join(visualization_path, "*_hdr_pred.exr")))
    hdr_gt_files = sorted(glob.glob(os.path.join(visualization_path, "*_hdr_gt.exr")))
    
    pred_peak_illuminance_list = []
    gt_peak_illuminance_list = []
    peak_error_list = []

    assert len(hdr_pred_files) == len(hdr_gt_files)
    for pred_file, gt_file in zip(hdr_pred_files, hdr_gt_files):
        pred = imageio.imread(pred_file)
        gt = imageio.imread(gt_file)

        pred_illuminance = 0.2126*pred[...,0] + 0.7152*pred[...,1] + 0.0722*pred[...,2]
        gt_illuminance = 0.2126*gt[...,0] + 0.7152*gt[...,1] + 0.0722*gt[...,2]

        pred_peak_illuminance = np.max(pred_illuminance)
        gt_peak_illuminance = np.max(gt_illuminance)

        if np.isinf(gt_peak_illuminance):
            continue
        
        # Order of magnitude comparsion
        pred_peak_illuminance_log10 = np.log10(pred_peak_illuminance).clip(0,100)
        gt_peak_illuminance_log10 = np.log10(gt_peak_illuminance).clip(0,100)

        peak_error = np.abs(pred_peak_illuminance_log10 - gt_peak_illuminance_log10) / gt_peak_illuminance_log10
        peak_error_list.append(peak_error)
    
    print(visualization_path)
    print(f"{colored('min: ', 'green')} {np.min(peak_error_list)}")
    print(f"{colored('max: ', 'green')} {np.max(peak_error_list)}")
    print(f"{colored('mean: ', 'green')} {np.mean(peak_error_list)}")
    print(f"{colored('median: ', 'green')} {np.median(peak_error_list)}")
    
    return np.min(peak_error_list), np.max(peak_error_list), np.mean(peak_error_list), np.median(peak_error_list)


def evaluate_peak_direction(visualization_path):
    """
    peak_direction_error_percentage : float
        angle of <pred_peak_dir, gt_peak_dir>

    Args: 
        visualization_path : str
            visulization is result on testset. 
            e.g. mc_to_sky/logs/Hold_Geoffroy_pred_hdr_pano_from_single/lightning_logs/version_0/visualization
            include 'xxxx_hdr_gt.exr', 'xxxx_hdr_pred.exr', ('xxxx_hdr_pred_rotated.exr')
    
    Returns:
        max, min, mean, median of peak_intensity_error_percentage
    """

    hdr_pred_files = sorted(glob.glob(os.path.join(visualization_path, "*_hdr_pred.exr")))
    hdr_gt_files = sorted(glob.glob(os.path.join(visualization_path, "*_ldr_input.png")))
    angular_error_list = []

    # Hold Geoffroy
    if len(glob.glob(os.path.join(visualization_path, "*_hdr_pred_rotated.exr"))) !=0:
        hdr_pred_files = sorted(glob.glob(os.path.join(visualization_path, "*_hdr_pred_rotated.exr")))

    assert len(hdr_pred_files) == len(hdr_gt_files)

    for pred_file, gt_file in zip(hdr_pred_files, hdr_gt_files):
        pred = imageio.imread(pred_file)
        gt = srgb_inv_gamma_correction(imageio.imread(gt_file) / 255)

        H, W, _ = pred.shape
        env_template = EnvironmentMap(H, 'skylatlong')

        pred_illuminance = 0.2126*pred[...,0] + 0.7152*pred[...,1] + 0.0722*pred[...,2]
        gt_illuminance = 0.2126*gt[...,0] + 0.7152*gt[...,1] + 0.0722*gt[...,2]

        max_index_pred = np.argmax(pred_illuminance, axis=None)
        max_index_pred_2d = np.unravel_index(max_index_pred, pred_illuminance.shape)
        peak_pred_v, peak_pred_u = max_index_pred_2d

        max_index_gt = np.argmax(gt_illuminance, axis=None)
        max_gt_illuminance = np.max(gt_illuminance)
        max_gt_illuminance_num = np.sum(gt_illuminance == max_gt_illuminance)

        if max_gt_illuminance_num > 15: 
            continue

        max_index_gt_2d = np.unravel_index(max_index_gt, gt_illuminance.shape)
        peak_gt_v, peak_gt_u = max_index_gt_2d

        # in sphere coordinate
        peak_pred_xyz = env_template.image2world(peak_pred_u / W, peak_pred_v / H)
        peak_gt_xyz = env_template.image2world(peak_gt_u / W, peak_gt_v / H)

        angular_error_cosine = np.dot(peak_pred_xyz / np.linalg.norm(peak_pred_xyz), peak_gt_xyz / np.linalg.norm(peak_gt_xyz))
        angular_error = np.degrees(np.arccos(angular_error_cosine))

        angular_error_list.append(angular_error)

    ic(len(angular_error_list))
    print(visualization_path)
    print(f"{colored('min: ', 'green')} {np.min(angular_error_list)}")
    print(f"{colored('max: ', 'green')} {np.max(angular_error_list)}")
    print(f"{colored('mean: ', 'green')} {np.mean(angular_error_list)}")
    print(f"{colored('median: ', 'green')} {np.median(angular_error_list)}")
    
    return np.min(angular_error_list), np.max(angular_error_list), np.mean(angular_error_list), np.median(angular_error_list)

if __name__ == "__main__":
    evaluate_peak_intensity("mc_to_sky/logs/Hold_Geoffroy_recon_pano_hdr_to_hdr_L1Loss/lightning_logs/version_0/visualization")
    evaluate_peak_intensity("mc_to_sky/logs/recon_pano_mc_to_sky/lightning_logs/version_0/visualization")
    evaluate_peak_intensity("mc_to_sky/logs/recon_pano_mc_to_sky_enhanced_white_balance_adjust_1104_231459/lightning_logs/version_0/visualization")

    ic('hold 1 view')
    evaluate_peak_direction("mc_to_sky/logs/Hold_Geoffroy_pred_hdr_pano_from_single/lightning_logs/version_0/visualization")
    ic('hold 3 views')
    evaluate_peak_direction("mc_to_sky/logs/rebuttal/hold-geoffroy-stage2-multiview-submission-version-pretrain_0126_130542/lightning_logs/version_0/visualization")
    ic('zian 1 view')
    evaluate_peak_direction("mc_to_sky/logs/pred_hdr_pano_from_single/lightning_logs/version_0/visualization")
    ic('zian 3 views')
    evaluate_peak_direction("mc_to_sky/logs/rebuttal/zian_stage2_multiviews_submission_version_pretrain_0126_130531/lightning_logs/version_0/visualization")
    ic('ours 3 views')
    evaluate_peak_direction("mc_to_sky/logs/pred_hdr_pano_from_AvgMultiView_enhanced_elu_white_balance_adjust3/lightning_logs/version_0/visualization")
