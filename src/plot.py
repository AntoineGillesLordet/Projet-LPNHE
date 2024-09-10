import matplotlib.pyplot as plt
import numpy as np
import corner
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from astropy.time import Time
import matplotlib.cm as cm
from matplotlib.colors import Normalize

color_band = {"ztfi":"black",
             "ztfr":"purple",
             "ztfg":"cyan",
             "megacam6::z":"blue",
             "megacam6::r":"red",
             "megacam6::g":"green",
             "megacam6::i2":"orange",
             }


def corner_(data, var_names=None, fig=None, labels=None, title=None, **kwargs):
    params = dict(
        var_names=var_names,
        show_titles=True,
        bins=50,
        smooth=0.9,
        label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=15),
        color="b",
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
        plot_density=False,
        plot_datapoints=True,
        fill_contours=True,
        max_n_ticks=7,
        hist_kwargs=dict(density=True, color=kwargs.get("color", "b")),
        labels=labels,
        alpha=0.2,
        fig=fig,
    )

    params.update(kwargs)

    fig = corner.corner(
        data,
        **params,
    )

    if title:
        fig.suptitle(title, fontsize=30)

    fig.set_dpi(50)
    return fig


def scatter_mollweide(data, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="mollweide")

    params = dict(s=1, color="k", alpha=0.3, marker=".")
    params.update(kwargs)
    ax.scatter(
        (data["ra"] - 360 * (data["ra"] > 180)) * np.pi / 180,
        data["dec"] * np.pi / 180,
        **params,
    )


def scatter_3d(x, y, z, bins=50):
    count, (binx, biny, binz) = np.histogramdd([x, y, z], bins=bins)
    count = gaussian_filter(count, 0.9)
    x_, y_, z_ = np.meshgrid(
        (binx[:-1] + binx[1:]) / 2 - binx.min(),
        (biny[:-1] + biny[1:]) / 2 - biny.min(),
        (binz[:-1] + binz[1:]) / 2 - binz.min(),
    )
    tree = cKDTree(np.dstack((x_.flatten(), y_.flatten(), z_.flatten())).reshape(-1, 3))
    pts = np.vstack([x - binx.min(), y - biny.min(), z - binz.min()]).T
    dist_id, nn_id = tree.query(pts, k=1)
    c = np.transpose(count, axes=[1, 0, 2]).flatten()[nn_id]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
    ax.scatter(x, y, z, c=c, s=0.5, alpha=0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.azim = -45
    ax.elev = 30

    cmap = cm.viridis
    norm = Normalize(vmin=c.min(), vmax=c.max())
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')

def plot_lc_index(index, lc_data, sne_data=None):
        
    for lc_nb in np.unique(lc_data[(lc_data.sn==index) & (lc_data.valid==1)].lc):
        lc = lc_data[(lc_data.lc==lc_nb) & (lc_data.valid==1)].sort_values(by='mjd')
        band = np.array(lc.band)[0]
        coef = 10 ** (-(lc["zp"] - 25) / 2.5)
        plt.errorbar(lc["mjd"],
                     lc["flux"]*coef,
                     yerr=lc["fluxerr"]*coef,
                     linestyle='',
                     marker='.',
                     color=color_band[band],
                     label=band
                    )
    if sne_data is not None:
        plt.axvline(sne_data.loc[index, "tmax"], color="purple", label=r"$t_0$")
        plt.axvline(sne_data.loc[index, "tmax"] - sne_data.loc[index, "err_tmax"], color="purple", linestyle=":")
        plt.axvline(sne_data.loc[index, "tmax"] + sne_data.loc[index, "err_tmax"], color="purple", linestyle=":")
    
    plt.title(f"SN {index}")
    plt.legend()
    
def _2D_cmap(x,y, renorm=True):
    xx, yy = x.copy(), y.copy()
    if renorm:
        xx -= xx.min()
        yy -= yy.min()
        xx /= xx.max()
        yy /= yy.max()
    return np.array(list(zip(0.7*((xx.ravel()-1)**2 + (yy.ravel()-(1-0.5/np.sqrt(2)))**2),
                               0.7*((xx.ravel())**2 + (yy.ravel()-(1-0.5/np.sqrt(2)))**2),
                               ((xx.ravel()-0.5)**2 + (yy.ravel()+0.5)**2)/2.5))
                   )

def plot_flux_phwl(input_flux, model_flux, wl_rf, ph_rf, linthresh=1e-3):
    fig, axs = plt.subplots(ncols=2, figsize=(20,10))
    axs[0].scatter(input_flux,
                   model_flux,
                   marker='.',
                   s=5,
                   alpha=0.5,
                   c=_2D_cmap(wl_rf, ph_rf, renorm=False)
                  )
    min_ = min(input_flux.min(), model_flux.min())
    max_ = max(input_flux.max(), model_flux.max())
    ext = max(np.abs(min_), np.abs(max_))
    axs[0].plot([-ext, ext], [-ext, ext], 'k:')
    axs[0].plot([-ext, ext], [ext, -ext], 'k:')
    
    if linthresh:
        axs[0].set_yscale('symlog', linthresh=linthresh)
        axs[0].set_xscale('symlog', linthresh=linthresh)
    
    axs[0].set_xlabel("Input Flux")
    axs[0].set_ylabel("Model Flux")

    wl_lin, ph_lin = np.meshgrid(np.linspace(2000, 9000, 1000), np.linspace(-20, 50, 1000))

    axs[1].pcolormesh(wl_lin, ph_lin, _2D_cmap(wl_lin, ph_lin).reshape((1000,1000,3)))
    axs[1].set_ylabel("Phase")
    axs[1].set_xlabel("Wavelength")
    axs[1].set_ylim((50, -20))
