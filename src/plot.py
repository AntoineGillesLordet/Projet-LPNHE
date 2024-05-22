import matplotlib.pyplot as plt
import numpy as np
import corner
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from astropy.time import Time


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
        linestyle="",
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
    ax.azim = -45
    ax.elev = 30



def plot_lc(dset, i, better_results=None, fig=None, **kwargs):
    _ = dset.show_target_lightcurve(index=i, s=8, fig=fig, **kwargs)
    plt.ylim(-200)
    target = dset.targets.data.loc[i]
    plt.axvline(Time(target["t0"], format="mjd").datetime, label=r'True $t_0$')
    if better_results:
        plt.axvline(Time(better_results.loc[i]["t0"], format="mjd").datetime, linestyle='--', c='darkblue', alpha=0.4, label = r'Fitted $t_0$')

        plt.axvline(Time(better_results.loc[i]["t0"] + better_results.loc[i]["err_t0"] , format="mjd").datetime, c='k', linestyle='dotted', label = r'$\sigma_{t_0}$')
        plt.axvline(Time(better_results.loc[i]["t0"] - better_results.loc[i]["err_t0"] , format="mjd").datetime, c='k', linestyle='dotted')

    
    plt.xlim(Time(target["t0"]-50, format="mjd").datetime, Time(target["t0"] +100, format="mjd").datetime)
    plt.legend()
    plt.title(f"Target {i}")
