import numpy as np
import pickle
from edris.tools import linear_interpolation_matrix
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

    results, meta, models = dset.fit_lightcurves(
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
    return results, meta, models


def run_edris(obs, cov, exp, **kwargs):
    logger.log(logging.INFO, "Initialisation of edris")
    interpol_matrix = linear_interpolation_matrix(
        jnp.log10(exp["z"]), jnp.log10(exp["z_bins"])
    )
    mu_start = jnp.linalg.solve(
        jnp.dot(interpol_matrix.T, interpol_matrix), jnp.dot(interpol_matrix.T, obs.mag)
    )
    x0 = {
        "mu_bins": mu_start,
        "coef": jnp.array([-0.14, 3.15]),
        "variables": obs.variables.reshape(-1, obs.mag.shape[0]),
        "sigma_int": 0.1,
    }

    L = lambda x: likelihood(
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
    
    return res, hessian(L)(res), loss, lmbda, iter_params
