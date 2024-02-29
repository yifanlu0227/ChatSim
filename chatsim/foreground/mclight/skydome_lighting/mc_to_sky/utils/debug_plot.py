"""
Use this tool to plot the feature map for verifying code.
"""

import matplotlib.pyplot as plt
import torch
import os

def plot_feature(feature, channel, save_path, flag="", vmin=None, vmax=None, colorbar=True):
    """
    Args:
        feature : torch.tensor or np.ndarry
            suppose in shape [N, C, H, W]

        channel : int or list of int
            channel for ploting

        save_path : str
            save path for visualizing results.
    """
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().cpu().numpy()

    if isinstance(channel, int):
        channel = [channel]
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    N, C, H, W = feature.shape
    for c in channel:
        for n in range(N):
            plt.imshow(feature[n,c], vmin=vmin, vmax=vmax)
            file_path = os.path.join(save_path, f"{flag}_agent_{n}_channel_{c}.png")
            if colorbar:
                plt.colorbar()
            plt.savefig(file_path, dpi=400)
            plt.close()
            print(f"Saving to {file_path}")