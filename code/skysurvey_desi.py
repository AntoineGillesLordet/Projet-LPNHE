import skysurvey
import matplotlib.pyplot as plt
import numpy as np
import corner
import pandas
import fitsio


def corner_(data, var_names=None, labels=None, title=None):
    fig = corner.corner(
        data,
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
        hist_kwargs=dict(density=True, color="b"),
        labels=labels,
        alpha=0.2,
    )

    if title:
        fig.suptitle(title, fontsize=30)
    
    fig.set_dpi(50)
    return fig


def load_bgs(path=None, columns=['RA', 'DEC', 'Z', 'Z_COSMO', 'STATUS']):
    if path is None:
        path = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/Uchuu/LightCone/BGS_v2/BGS_LC_Uchuu.fits"

    data = fitsio.read(path, columns=columns)

    df = pandas.DataFrame(data)
    df.rename(columns={'RA':'ra', 'DEC':'dec','Z':'z', 'Z_COSMO':'z_cosmo'}, inplace=True)
    for col in df.columns:
        if (df[col].dtype == np.dtype('>f8') or df[col].dtype == np.dtype('>f4')):
            df[col] = np.float64(df[col])
        elif df[col].dtype == np.dtype('>i4'):
            df[col] = np.int64(df[col])

    if 'STATUS' in df.columns:
        df['in_desi'] = df['STATUS'] & 2**1 != 0
        df.drop(columns=['STATUS'], inplace=True)

    return df


def draw_SN(size=10000, **bgs_kwargs):
    snia = skysurvey.SNeIa()
    data = snia.draw(size=size, inplace=True)
    
    bgs_data = load_bgs(**bgs_kwargs)
    indexes = np.random.choice(bgs_data[bgs_data['in_desi']].index, size=size)
    
    snia.data['ra'] = bgs_data.loc[indexes].reset_index()['ra']
    snia.data['dec'] = bgs_data.loc[indexes].reset_index()['dec']
    snia.data['z'] = bgs_data.loc[indexes].reset_index()['z']
    
    return snia