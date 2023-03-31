import matplotlib.pyplot as plt
import json
import scipy.ndimage as ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib as mpl
import numpy as np

def plot_from_density(density : np.ndarray, vmin : float =0., vmax : float =1., fig = None, ax = None, colorbar=True, mask_vmin=-float('inf'), mask_vmax=float('inf'), sigma=0.0) -> None:
    """
    Usual values of mask vmax and vmin : 1e-3, 1e3
    NOTE : We transpose the matrix for whatever reason, don't change this as it will change some experiments
    Consequence : row => columns => x-axis
                  columns => rows => y-axis
    """
    N = len(density)
    
    
    density = density.T

    masked = ndimage.gaussian_filter(density, sigma=sigma)
    masked = np.ma.masked_outside(masked, mask_vmin, mask_vmax)
    if ax is None:
        plt.clf()
        plt.imshow(masked, extent=(vmin, vmax, vmin, vmax), origin='lower', norm=LogNorm())
        ax = plt.gca()
        ax.set_facecolor((0., 0., 0.))
        norm = mpl.colors.LogNorm(vmin=mask_vmin, vmax=mask_vmax)
        if colorbar:
            cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm))
        # cbar.set_ticklabels(["1e-7", "1e-5", "1e-3", "1e-1", "1e1"])

    else:
        im = ax.imshow(masked, extent=(vmin, vmax, vmin, vmax), origin='lower', norm=LogNorm())
        ax.set_facecolor((0., 0., 0.))
        return im
