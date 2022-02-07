import matplotlib.pyplot as plt
import json
import scipy.ndimage as ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib as mpl
import numpy as np

import datalogging

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


def plot_from_file(file : str) -> np.ndarray:   
    configuration, results = datalogging.load_config_and_result(file)
    info = configuration | results

    # row is BO i.e y axis, column (i.e x axis) is ERM
    vmin    = info['vmin'] if 'vmin' in info else 0.
    vmax    = info['vmax'] if 'vmax' in info else 1.
    density = np.array(info['density'])
    N       = len(density)
    # ERM becomes y-axis (=indexed by row), BO becomes x-axis (=columns)
    
    plot_from_density(density, vmin, vmax)

    type = info['type'] if 'type' in info else ''
    alpha = info['alpha'] if 'alpha' in info else info['alpha_range']
    
    plt.title('$\\alpha = $ {:.2f}'.format(alpha) + ', $\\lambda = $ {:.3f}'.format(info['lamb']) + type)
    
    return density

def plot_surface_from_file(file : str, stride : int =10) -> np.ndarray:   
    configuration, results = datalogging.load_config_and_result(file)
    info = configuration | results

    # row is BO i.e y axis, column (i.e x axis) is ERM
    N = info['N']
    density = np.array(info['density'])
    # ERM becomes y-axis (=indexed by row), BO becomes x-axis (=columns)
    density = density.T
    
    vmin    = info['vmin'] if 'vmin' in info else 0.
    vmax    = info['vmax'] if 'vmax' in info else 1.

    plt.clf()
    X = np.linspace(vmin, vmax, N)
    Y = np.copy(X)
    X, Y = np.meshgrid(X, Y)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.zaxis.set_scale('log')
    ax.plot_wireframe(X, Y, density, rstride=stride, cstride=stride)

    alpha = info['alpha'] if 'alpha' in info else info['alpha_range']
    plt.title('$\\alpha = $ {:.2f}'.format(alpha) + ', $\\lambda = $ {:.3f}'.format(info['lamb']))

    return density