import matplotlib.pyplot as plt
import numpy as np
import corner
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from astropy.time import Time
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from numpy.lib.histograms import _get_bin_edges
import jax.numpy as jnp
import healpy

from .tools import wrapp_around

color_band = {"ztfi":"olive",
             "ztfr":"purple",
             "ztfg":"limegreen",
             "ztf::i":"olive",
             "ztf::r":"purple",
             "ztf::g":"limegreen",
             "megacam6::z":"blue",
             "megacam6::r":"red",
             "megacam6::g":"green",
             "megacam6::i2":"orange",
             "megacam6::i":"orange",
             "hsc::Y":"turquoise",
             "hsc::g":"olivedrab",
             "hsc::i2":"gray", 
             "hsc::r2":"darkred",
             "hsc::z":"gold",
             }

def set_rc():
    """
    Set some matplotlib rc params to get clean plots for papers
    """
    plt.rc('axes', labelsize=16, linewidth=2.)
    plt.rc('xtick', labelsize=14)
    plt.rc('xtick.major', width=2)
    plt.rc('ytick', labelsize=14)
    plt.rc('ytick.major', width=2)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titleweight='bold')

def corner_(data, var_names=None, labels=None, fig=None, title=None, return_fig=False, **kwargs):
    """
    Wrapper around ``corner.corner()`` for quick plotting of a dataset

    Parameters
    ----------
    data : pandas.Dataframe or np.ndarray
        Data to plot, see ``corner`` documentation for more details.
    var_names : str list, optional
        In case of a Dataframe, columns to use in the corner plot.
    labels : str list, optional
        Labels of the different axes. If not provided, defaults to ``var_names``.
    fig : matplotlib.Figure, optional
        Figure to plot on, useful for overplotting different datasets.
    title : str, optional
        Title of the plot.
    return_fig : bool, optional
        Wether to return the figure, should be used with ``fig`` for plotting different datasets.
    **kwargs : Any
        All kwargs are passed to ``corner.corner``.
    """
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
    if return_fig:
        return fig


def scatter_mollweide(data, ax=None, unit="degree", rot=(0,0), **kwargs):
    """
    Wrapper around ``matplotlib.scatter`` for quick plotting of a dataset in mollweide projection

    Parameters
    ----------
    data : pandas.Dataframe
        Data to plot, should contain columns labeled ``["ra", "dec"]``.
    ax : matplotlib.projection.geo.MollweideAxes, optional
        MollweideAxes instance to plot on, if not provided creates a new figure.
    unit : str, optional
        Wether given coordinates are in 'degree' or 'rad'. Default is 'degree',
    **kwargs : Any
        All kwargs are passed to ``matplotlib.scatter``.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="mollweide")

    params = dict(s=1, alpha=0.3, marker=".")
    params.update(kwargs)

    ra, dec = data["ra"] - rot[0], data["dec"] - rot[1]

    return ax.scatter(
            *wrapp_around(ra,dec, unit_in=unit, unit_out='rad'),
            **params,
        )
    
def add_clusters(clusters, ax=None, color="tab:red"):
    """
    Scatter + labels for a cluster list
    """
    if ax==None:
        ax=plt.gca()    
    scatter_mollweide(clusters, ax=ax, marker='*', color=color, s=50, alpha=1)
    for name, ra, dec in clusters[["name", "ra", "dec"]].values:
        ax.text(ra*np.pi/180 - 2*np.pi*(ra > 180) +0.03, dec*np.pi/180 +0.03, name, fontsize=7, color=color)

def add_skymap(ax, skymap, cmap='viridis', colorbar=True, location='bottom', shrink=0.8, pad=0.05, aspect=50, cb_label=None, rm_ticks=True,
               plottype='pcolormesh',
               **kwargs):
    """
    Skymap handling as a matplotlib pcolormesh instead of the weird shenaningans healpy does to axes and figures.
    """
    kwargs_projview = dict(flip="geo", projection_type="mollweide")
    kwargs_projview.update({k:v for k, v in kwargs.items() if k in healpy.projview.__code__.co_varnames})
    other_kwargs={k:v for k, v in kwargs.items() if (k not in healpy.projview.__code__.co_varnames) or (k == 'norm')}
    ra_, dec_, map_ = healpy.projview(skymap, return_only_data=True, **kwargs_projview)
    try:
        # Cartopy support
        from cartopy.mpl.geoaxes import GeoAxes
        if isinstance(ax, GeoAxes):
            ra_*=180/np.pi
            dec_*=180/np.pi
    except:
        pass
    if plottype=="pcolormesh":
        pmesh = ax.pcolormesh(ra_, dec_, map_, cmap=cmap, rasterized=True, **other_kwargs)
    elif plottype=="contourf":
        pmesh = ax.contourf(ra_, dec_, map_, cmap=cmap, **other_kwargs)
    elif plottype=="contour":
        pmesh = ax.contour(ra_, dec_, map_, cmap=cmap, **other_kwargs)

    if colorbar:
        colorbar_kw = dict(location=location,
                           shrink=shrink,
                           pad=pad,
                           aspect=aspect)
        cb = plt.colorbar(pmesh, ax=ax, **colorbar_kw)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(label=cb_label, fontsize=12)
    if rm_ticks:    
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return pmesh

def add_grid(ax, t_step=30, p_step=60, rot=[0], **kwargs):
    """
    Graticule for mollweide projection (stolen from healpy)
    """
    default_style = dict(linewidth = 0.75, linestyle="-", color="grey", alpha=0.8)
    default_style.update(**kwargs)

    rotated_grid_lines, _ = healpy.newvisufunc.CreateRotatedGraticule(rot=rot, t_step=t_step, p_step=p_step)

    for g_line in rotated_grid_lines:
        ax.plot(*g_line, **default_style)
    thetaSpacing = np.arange(-90, 90 + t_step, t_step)
    phiSpacing = np.arange(-180, 180 + p_step, p_step)
    ax.set_yticks(thetaSpacing[1:-1]*np.pi/180, list(map(lambda x: f'{x:0d}°', thetaSpacing[1:-1])))
    ax.set_xticks(phiSpacing[1:-1]*np.pi/180, list(map(lambda x: f'{x+360*(x < 0):0d}°', phiSpacing[1:-1])))
    ax.tick_params(labelsize=7)



def scatter_3d(x, y, z, bins=50):
    """
    3D scatter plot in cartesian coordinates with color coding according to the local density of points

    Parameters
    ----------
    x : array-like, shape (n, )
        X coordinates.
    y : array-like, shape (n, )
        Y coordinates.
    z : array-like, shape (n, )
        Z coordinates.
    bins : int, optional
        Number of bins for the density evaluation.
    """

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

    


def plot_res(data, data_truth, col, label, unit=None, log=False, linthresh=1e-3):
    """
    Plot the residuals of the data with respect to the truth

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the reconstructed values.
    data_truth : pandas.DataFrame
        Dataframe containing the true values.
    col : str
        Name of the column to use.
    label : str
        Label to use in the axes/title.
    unit : str, optional
        Unit to add in the labels, default labels have no units.
    log : bool, optional
        Wether to plot with a symmetric log scale.
    linthresh : float, optional
        In the case of a symmetric log scale, threshold for the linear scale around 0.
    """
    fig, axs = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row', figsize=(10,7))
    axs[0,0].errorbar(data_truth[col], data[col],
                      yerr = data["err_"+col] if "err_"+col in data.columns else None,
                      linestyle="", marker=".")
    
    axs[0,0].plot([data_truth[col].min() - 1e-3, data_truth[col].max() + 1e-3], [data_truth[col].min() - 1e-3, data_truth[col].max() + 1e-3], linestyle=':', color='r')
    delta = data_truth[col] - data[col] 
    axs[1,0].errorbar(data_truth[col], delta,
                      yerr = data["err_"+col] if "err_"+col in data.columns else None,
                      linestyle="", marker=".")
    axs[1,0].axhline(0, linestyle=':', color='r')

    if log:
        axs[0,0].loglog()
        axs[1,0].semilogx()
        axs[1,0].set_yscale("symlog", linthresh=linthresh)
        
        l_bins = np.logspace(np.log10(np.min(np.abs(delta))) - 1e-10, np.log10(np.max(np.abs(delta))) + 1e-10, 40)
        _=axs[1,1].hist(delta, bins=[*(-l_bins[::-1]), *l_bins], orientation='horizontal')
    else:
        _=axs[1,1].hist(delta, bins=80, orientation='horizontal')
    axs[1,1].set_xlabel("Count")
    mean, std = delta.mean(), delta.std()
    axs[1,1].axhline(mean, linestyle='--', color='tab:purple', label=f'Mean : {mean:.3e}')
    axs[1,1].axhline(mean-std, linestyle=':', color='tab:purple', label=f'Std : {std:.3e}')
    axs[1,1].axhline(mean+std, linestyle=':', color='tab:purple')
    axs[1,1].legend()

    if unit:
        axs[0,0].set_xlabel(label.lower() + '$_{Truth}$ (' + unit + ')')
        axs[0,0].set_ylabel(label + ' (' + unit + ')')
        axs[1,0].set_xlabel(label.lower() + '$_{Truth}$ (' + unit + ')')
        axs[1,0].set_ylabel(label.lower() + '$_{Truth} - $' + ' (' + unit + ')')
    else:
        axs[0,0].set_xlabel(label.lower() + '$_{Truth}$')
        axs[0,0].set_ylabel(label)
        axs[1,0].set_xlabel(label.lower() + '$_{Truth}$')
        axs[1,0].set_ylabel(label.lower() + '$_{Truth} - $' + label)

    fig.delaxes(axs[0,1])


potential_keys = ['H0', 'M0', 'Omega_m', 'Omega_r', 'Omega_l', 'coef', 'sigma_int']
latex_keys = {'H0': '$H_0$',
              'M0': '$M_0$',
              'Omega_m': '$\\Omega_m$',
              'Omega_r': '$\\Omega_r$',
              'Omega_l': '$\\Omega_l$',
              'coef': ['$\\alpha$', '$\\beta$'],
              'sigma_int': '$\\sigma_{int}$'}
    
def plot_edris_biais(res, x0, cov_res):
    """
    Plot for the bias in the edris parameters, could be enhanced
    
    Parameters
    ----------
    res : dict
        Edris result
    x0 : dict
        True values
    cov_res : dict
        Edris output covariance
    """
    fig, ax = plt.subplots(figsize=(5,5))
    keys = np.array(list(res.keys()))[[p in potential_keys for p in res.keys()]]
    labels, values, diffs = [], [], []
    for k in keys:
        labels += latex_keys[k] if isinstance(latex_keys[k], list) else [latex_keys[k]]
        values += list(res[k])
        diffs +=  list(res[k] - x0[k])
    n_pars = len(keys)
    ax.hlines(0, -1, 5*n_pars+1, color='r', linestyle=':')
    ax.set_xticks(5*np.arange(n_pars+1), labels)
    ax.set_ylabel('Deviation')
    plt.errorbar(5*np.arange(n_pars+1), diffs,
                 yerr= jnp.sqrt(jnp.diag(cov_res)[:n_pars+1]),
                 linestyle='',
                 marker='.',
                 capsize=5,
                 capthick=.5)
    fig, ax = plt.subplots(figsize=(1,1))
    ax.axis("off")
    for i, pos in enumerate(np.arange(n_pars+1)*.4):
        fig.text(0, pos, labels[i] + f" = {values[i]:.3f} $\\pm$ {jnp.sqrt(jnp.diag(cov_res)[i]):.3f}")


def plot_hubble(obs, exp, res, cov_res, cosmo, x0):
    """
    Hubble diagram for edris output and comparison between true and reconstructed cosmology.
    
    Parameters
    ----------
    obs : edris.Obs
        Observations containing magnitudes and some standardisation variables
    exp : dict
        Explanatory variables. Contains the redshifts of the SN in ``z`` and the redshift bins in ``z_bins``.
    res : dict
        Edris results
    cov_res : dict
        Edris output covariance
    cosmo : func
        Cosmology as ``cosmo(x0, dict(z=[...]))``, as used by edris.
    x0 : dict
        True values
    """
    std_mag = obs.mag - jnp.matmul(res["coef"], res["variables"])

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex="col", figsize=(7, 6), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.set_xscale('log')
    ax1.scatter(
        exp["z"], std_mag, s=0.5, alpha=0.3, color="k", label="Standardised magnitudes"
    )
    ax1.plot(
        np.linspace(5e-3, 0.8, 1000),
        cosmo(x0, {'z':np.linspace(5e-3, 0.8, 1000)}),
        color="tab:green",
        linestyle=":",
        label="Underlying cosmo",
    )

    ax2.scatter(exp["z"], std_mag - cosmo(x0, exp), color="k", s=0.5, alpha=0.3)
    
    if "mu_bins" in res.keys():
        ax1.errorbar(
            exp["z_bins"],
            res["mu_bins"],
            yerr=jnp.sqrt(jnp.diag(cov_res[2 : 2 + n_bins, 2 : 2 + n_bins])),
            color="tab:blue",
            label="edris",
        )
        ax2.errorbar(
            exp["z_bins"],
            res["mu_bins"] - cosmo(x0, {"z" : exp["z_bins"]}),
            yerr=jnp.sqrt(jnp.diag(cov_res[2 : 2 + n_bins, 2 : 2 + n_bins])),
            color="tab:blue",
        )
    else:
        ax1.plot(
            jnp.linspace(5e-3, 0.8, 1000),
            cosmo(res, {"z": jnp.linspace(5e-3, 0.8, 1000)}),
            color="tab:blue",
            label="Edris fit",
        )
        ax2.plot(
            jnp.linspace(5e-3, 0.8, 1000),
            cosmo(res, {"z": jnp.linspace(5e-3, 0.8, 1000)}) - cosmo(x0, {"z":jnp.linspace(5e-3, 0.8, 1000)}),
            color="tab:blue",
        )
    
    
    ax1.legend()
    ax1.set_ylabel(r"$\mu$")
    
    lims = ax2.get_xlim()
    ax2.hlines(0.0, xmin=lims[0], xmax=lims[1], color="tab:green", linestyle=":")
    ax2.set_xlim(*lims)
    ax2.set_ylabel(r"$\Delta\mu$")
    ax2.set_xlabel(r"$z$")
    
    fig.suptitle(r"Edris fitted model")


def plot_binned_data(to_bin, data, bins, **kwargs):
    """
    Nice utility function for plotting mean and stds of data in bins.

    Parameters
    ----------
    to_bin : array-like, shape (n,)
        Values to bin
    data : array-like, shape (n,)
        Values to average
    bins : int or array-like
        Bins to use, similar to ``numpy.histogram`` parameter ``bins``.
    **kwargs : Any, optional
        All kwargs are passed to ``matplotlib.pyplot.errorbar``.

    """
    bin_edges, uniform_bins = _get_bin_edges(to_bin, bins, None, None)
    idx = to_bin.argsort()
    to_bin = to_bin[idx]
    data = data[idx]
    bounds =np.concatenate((
        to_bin.searchsorted(bin_edges[:-1], 'left'),
        to_bin.searchsorted(bin_edges[-1:], 'right')))
    means = [data[start:stop].mean() for start, stop in zip(bounds[:-1],bounds[1:])]
    stds = [data[start:stop].std() for start, stop in zip(bounds[:-1],bounds[1:])]
    
    default_style = dict(linestyle='',
             linewidth=1,
             capsize=2)
    default_style.update(**kwargs)
    plt.errorbar((bin_edges[:-1]+bin_edges[1:])/2, means, stds, (bin_edges[1:] - bin_edges[:-1])/2, **default_style)


def plot_lc_index(index, lc_data, sne_data=None):
    """
    My plotting for lightcurve because I was tired of having to put everything in a skysurvey dataset.
    """
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
        if err_tmax in sne_data.columns:
            plt.axvline(sne_data.loc[index, "tmax"] - sne_data.loc[index, "err_tmax"], color="purple", linestyle=":")
            plt.axvline(sne_data.loc[index, "tmax"] + sne_data.loc[index, "err_tmax"], color="purple", linestyle=":")
    
    plt.title(f"SN {index}")
    plt.legend()
