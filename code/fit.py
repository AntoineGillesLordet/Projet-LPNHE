import numpy as np
import pickle
from edris.tools import linear_interpolation_matrix, restrict
from edris.models import likelihood, binned_cosmo
from edris.minimize import tncg
import jax.numpy as jnp
from jax import hessian
from .logging import logging, logger
import sncosmo

try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x: x


def fit_lc(dset, index, savefile=None, **kwargs):
    """
    Fit lightcurves using the SALT2 model with skysurvey/sncosmo using the following bounds:
    * ``z`` is fixed
    * ``t0`` is fitted in ``[t0_true-20 ; t0_true+30]``
    * ``c`` is fitted in ``[-0.8;1.0]``
    * ``x0`` is fitted in ``[-0.8;0.8]``
    * ``x1`` is fitted in ``[-6.0;6.0]``
    
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
    data = dset.targets.data.loc[index].copy()
    fixed = {"z": data["z"]}

    guess = {
        "t0": data["t0"],
        "c": data["c"],
        "x0": data["x0"],
        "x1": data["x1"],
    }
    bounds = {
        "t0": data["t0"].apply(lambda x: [x - 20, x + 30]),
        "c": data["c"].apply(lambda x: [-0.8, 1.0]),
        "x0": data["x0"].apply(lambda x: [-0.8, 0.8]),
        "x1": data["x1"].apply(lambda x: [-6, 6]),
    }

    params = dict(phase_fitrange=[-40,130], maxcall=10000)
    params.update(kwargs)

    results, meta = dset.fit_lightcurves(
        source=sncosmo.Model("salt2"),
        index=index,
        use_dask=False,
        fixedparams=fixed,
        guessparams=guess,
        bounds=bounds,
        **params
    )

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
          'mu_bins':jnp.zeros(len(exp['z_bins'])),
          'coef':jnp.array([-0.14,3.15]),
          'variables':jnp.array(obs.variables.reshape((2,-1 ))),
         }
    delta_mu = obs.mag - edris.models.sn1a_model(x0, exp).mag
    interpol_matrix = edris.tools.linear_interpolation_matrix(jnp.log10(exp['z']), jnp.log10(exp['z_bins']))
    mu_start = jnp.linalg.solve(jnp.dot(interpol_matrix.T, interpol_matrix), jnp.dot(interpol_matrix.T, delta_mu))
    x0['mu_bins'] = mu_start
    
    L = lambda x: restrict(likelihood, {'sigma_int':0.1,})(
        x, exp, cov, obs, cosmo=binned_cosmo, truncated=False, restricted=False
    )
    params = dict(niter=1000,
                  lmbda=1e4,
                  tol=1e-2,
                  max_iter_tncg=None)
    params.update(kwargs)

    logger.log(logging.INFO, f"Running edris with parameters {kwargs}")
    res, loss, lmbda, iter_params = tncg(L, x0, **params)
    logger.log(logging.INFO, "Done")
    
    return res, hessian(L)(res), loss, iter_params
