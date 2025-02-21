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
from scipy.optimize import curve_fit

try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x: x

from astropy.table import Table

def fit_sn(model, sn, lc, data, modelcov=True):
    from iminuit.warnings import IMinuitWarning
    import warnings
    warnings.filterwarnings("ignore", category=IMinuitWarning)
    try:
        lc_sn = lc.loc[sn]
        lc_sncosmo=Table.from_pandas(lc_sn[['sn','name','mjd','flux','fluxerr','magsys','exptime','valid','lc','band','mag_sky','seeing','zp']])

        model.set(z=data.loc[sn, "z"],
                  mwebv=data.loc[sn, 'mwebv'], mwr_v=3.1)  # set the model's redshift and MW
        t0=data.loc[sn, 'tmax']
        res, _ = sncosmo.fit_lc(lc_sncosmo, model, ['t0', 'x0', 'x1', 'c'],
                                bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3), 't0':(t0-1, t0+1)}, phase_range=None, modelcov=modelcov)
        return res
    except:
        return None

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


def fit_cosmo(z_bins, mu_bins, cov, cosmo):
    logger.log(logging.INFO, "Fitting cosmology to edris result")
    def dist(z, Omega_r, Omega_m, Omega_l, H0):
        Omega_k = 1. - Omega_m - Omega_l - Omega_r
        return quad(lambda z1 : (Omega_m*(1+z1)**3 + Omega_r*(1+z1)**4 + Omega_k*(1+z1)**2 + Omega_l)**(-0.5)*c.value*10**(-3)/H0, 0, z)

    dist_vec = np.vectorize(dist)

    def z_to_mag(z, Omega_m, Mb=25-19.3, Omega_r=cosmo.Ogamma0 + cosmo.Onu0, Omega_l=cosmo.Ode0, H0=cosmo.H0.value):
        return 5.0 * np.log10(abs((z + 1.0) * dist_vec(z, Omega_r, Omega_m, Omega_l, H0)[0])) + Mb

    popt, pcov, = curve_fit(z_to_mag,
                           z_bins,
                           mu_bins,
                           sigma=cov,
                           p0=[0.3],
                           bounds=([0.],[1.]),
                          )
    mag_to_z_cosmo = jnp.vectorize(interp1d(z_to_mag(np.linspace(1e-6, 0.1, 10000), *popt), np.linspace(1e-6, 0.1,10000)), signature='(k)->(k)')

    return popt, pcov, mag_to_z_cosmo, z_to_mag