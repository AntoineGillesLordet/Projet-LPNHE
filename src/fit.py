import numpy as np
import pickle
from edris.tools import linear_interpolation_matrix, restrict
from edris.models import likelihood, binned_cosmo, sn1a_model
from edris.minimize import tncg
import jax.numpy as jnp
from jax import hessian
from .logging import logging, logger
import sncosmo
from scipy.interpolate import interp1d
from astropy.constants import c
from scipy.integrate import quad
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import curve_fit

try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x: x


def fit_lc(dset, index, savefile=None, **kwargs):
    """
    Fit lightcurves using the SALT2 model with skysurvey/sncosmo using the following bounds:
    * ``z`` is fixed
    * ``t0`` is fitted in ``[t0_true-15 ; t0_true+30]``
    * ``c`` is fitted in ``[-3;3]``
    * ``x0`` is fitted in ``[-0.1;10]``
    * ``x1`` is fitted in ``[-5;5]``

    Parameters
    ----------
    dset : skysurvey.DataSet
        Dataset containing the targets and lightcurves.
    index : np.ndarray or list
        Indexes of the targets to fit.
    savefile : str, optional
        File to pickle the lightcurves, targets and results to.
        Default ``None`` means not saving.
    **kwargs : Any
        All kwargs are passed to ``dset.fit_lightcurves``.

    Return
    ------
    results : pandas.Dataframe
        Converged points and covariance matrices as a dataframe
    meta : dict list
        Metadata of the fits, the key ``success`` flags fits that converged
    """

    logger.log(logging.INFO, f"Running LC fit")
    fixed = {
        "z": dset.targets.data.loc[index, "z"],
        "mwebv": dset.targets.data.loc[index, "mwebv"],
        "mwr_v": [3.1]*len(index),
    }

    guess = {
        "t0": dset.targets.data.loc[index, "t0"],
        "c": dset.targets.data.loc[index, "c"],
        "x0": dset.targets.data.loc[index, "x0"],
        "x1": dset.targets.data.loc[index, "x1"],
    }
    bounds = {
        "t0": dset.targets.data.loc[index, "t0"].apply(lambda x : [x-15, x+30]),
        "c": [[-3, 3]]*len(index),
        "x0": [[-.1, 10]]*len(index),
        "x1": [[-5, 5]]*len(index),
    }

    params = dict(phase_fitrange=[-40, 80], maxcall=10000, modelcov=True)
    params.update(kwargs)

    results, meta = dset.fit_lightcurves(
        source=dset.targets._template._sncosmo_model,
        index=index,
        use_dask=False,
        fixedparams=fixed,
        guessparams=guess,
        bounds=bounds,
        **params,
    )
    dset.targets.data['converged'] = False

    for i in index:
        dset.targets.data.loc[i,'converged'] = meta[(i,"success")]

    if savefile:
        logger.log(logging.INFO, "Saving")
        with open(savefile, "wb") as f:
            pickle.dump(dset.data, f)
            pickle.dump(dset.targets.data, f)
            pickle.dump(results, f)
            pickle.dump(meta, f)
            logger.log(logging.INFO, "Done")
    return results, meta


def run_edris(obs, cov, exp, **kwargs):
    """
    Computes the starting point and run edris

    Parameters
    ----------
    obs : edris.Obs
        Observations containing magnitudes and some standardisation variables
    cov : edris.CovMatrix
        Covariance matrix of the previous parameters (magnitudes and standardisation variables)
    exp : dict
        Explanatory variables. Should contain the redshifts of the SN in ``z`` and the redshift bins in ``z_bins``.

    Return
    ------
    res : dict
        Result of the fit
    hessian : dict
        Hessian of the likelihood at the result point
    loss : list
        List of the loss values during the fit
    iter_params : list
        Parameters at each iterations
    """

    logger.log(logging.INFO, "Computing starting point")

    x0 = {
        "mu_bins": jnp.zeros(len(exp["z_bins"])),
        "coef": jnp.array([-0.14, 3.15]),
        "variables": jnp.array(obs.variables.reshape((2, -1))),
    }
    delta_mu = obs.mag - sn1a_model(x0, exp).mag
    interpol_matrix = linear_interpolation_matrix(
        jnp.log10(exp["z"]), jnp.log10(exp["z_bins"])
    )
    mu_start = jnp.linalg.solve(
        jnp.dot(interpol_matrix.T, interpol_matrix),
        jnp.dot(interpol_matrix.T, delta_mu),
    )
    x0["mu_bins"] = mu_start

    L = lambda x: restrict(
        likelihood,
        {
            "sigma_int": 0.1,
        },
    )(x, exp, cov, obs, cosmo=binned_cosmo, truncated=False, restricted=False)
    params = dict(niter=1000, lmbda=1e4, tol=1e-2, max_iter_tncg=None)
    params.update(kwargs)

    logger.log(logging.INFO, f"Running edris with parameters {params}")
    res, loss, lmbda, iter_params = tncg(L, x0, **params)
    logger.log(logging.INFO, "Done")

    return res, hessian(L)(res), loss, iter_params


def fit_cosmo(z_bins, mu_bins, cov):
    logger.log(logging.INFO, "Fitting cosmology to edris result")
    def dist(z, Omega_r, Omega_m, Omega_l, H0):
        Omega_k = 1. - Omega_m - Omega_l - Omega_r
        return quad(lambda z1 : (Omega_m*(1+z1)**3 + Omega_r*(1+z1)**4 + Omega_k*(1+z1)**2 + Omega_l)**(-0.5)*c.value*10**(-3)/H0, 0, z)

    dist_vec = np.vectorize(dist)

    def z_to_mag(z, Omega_m, Mb, Omega_r=cosmo.Ogamma0 + cosmo.Onu0, Omega_l=cosmo.Ode0, H0=cosmo.H0.value):
        return 5.0 * np.log10(abs((z + 1.0) * dist_vec(z, Omega_r, Omega_m, Omega_l, H0)[0])) + Mb

    popt, pcov, = curve_fit(z_to_mag,
                           z_bins,
                           mu_bins,
                           sigma=cov,
                           p0=[0.3, 25-19.3],
                           bounds=([0., 0.],[1., ]),
                          )
    mag_to_z_cosmo = jnp.vectorize(interp1d(z_to_mag(np.linspace(1e-6, 0.1, 10000), *popt), np.linspace(1e-6, 0.1,10000)), signature='(k)->(k)')

    return popt, pcov, mag_to_z_cosmo, z_to_mag