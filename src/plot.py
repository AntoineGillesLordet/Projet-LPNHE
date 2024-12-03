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


def plot_res(data, data_truth, col, label, unit=None, log=False, linthresh=1e-3):
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
    # plt.savefig('../figures/Uchuu_final_params.png')
    fig, ax = plt.subplots(figsize=(1,1))
    ax.axis("off")
    for i, pos in enumerate(np.arange(n_pars+1)*.4):
        fig.text(0, pos, labels[i] + f" = {values[i]:.3f} $\\pm$ {jnp.sqrt(jnp.diag(cov_res)[i]):.3f}")
        # fig.text(0, .4, f"$\\Omega_l = ${res['Omega_l'][0]:.3f} $\\pm$ {jnp.sqrt(jnp.diag(cov_res)[1]):.3f}")
        # fig.text(0, 0., f"$\\alpha = ${res['coef'][0]:.3f} $\\pm$ {jnp.sqrt(jnp.diag(cov_res)[2]):.3f}")
        # fig.text(0, -.4, f"$\\beta = ${res['coef'][1]:.3f} $\\pm$ {jnp.sqrt(jnp.diag(cov_res)[3]):.3f}")


