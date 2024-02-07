import matplotlib.pyplot as plt
import numpy as np
import corner

def corner_(data, var_names=None, fig=None, labels=None, title=None, **kwargs):
    params = dict(var_names=var_names,
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
        hist_kwargs=dict(density=True, color=kwargs.get("color",'b')),
        labels=labels,
        alpha=0.2,
        fig = fig)
    
    params.update(kwargs)

    fig = corner.corner(
        data,
        **params,
    )

    if title:
        fig.suptitle(title, fontsize=30)
    
    fig.set_dpi(50)
    return fig

def mollweide_scatter(data, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="mollweide")

    params=dict(s=1, color='k',alpha=0.3, marker=".")
    params.update(kwargs)
    ax.scatter(
        (data["ra"] - 360 * (data["ra"] > 180)) * np.pi / 180,
        data["dec"] * np.pi / 180,
        linestyle="",
        **params,
    )