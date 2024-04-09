import numpy as np
import pickle
from edris.tools import linear_interpolation_matrix
from edris.models import likelihood, binned_cosmo
from edris.minimize import tncg
import jax.numpy as jnp
import sncosmo

try:
    from tqdm import tqdm
except:
    tqdm = lambda x:x    


def fit_lc(dset, index, savefile=None, **kwargs):
    fixed = {"z": dset.targets.data.loc[index]["z"]}

    guess = {
        "t0": dset.targets.data.loc[index]["t0"],
        "c": dset.targets.data.loc[index]["c"],
        "x0": dset.targets.data.loc[index]["x0"],
        "x1": dset.targets.data.loc[index]["x1"],
    }
    bounds = {
        "t0": dset.targets.data.loc[index]["t0"].apply(lambda x: [x-20, x+30]),
        "c": dset.targets.data.loc[index]["c"].apply(lambda x: [-0.3, 1.0]),
        "x0": dset.targets.data.loc[index]["x0"].apply(lambda x: [-0.1, 0.1]),
        "x1": dset.targets.data.loc[index]["x1"].apply(lambda x: [-4, 4]),
    }

    params=dict(phase_fitrange=[-40,130],
                maxcall=10000)
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
        with open(savefile, 'wb') as f:
            pickle.dump(dset.data, f)
            pickle.dump(dset.targets.data, f)
            pickle.dump(results, f)

    return results


def run_edris(obs, cov, exp, **kwargs):
    interpol_matrix = linear_interpolation_matrix(jnp.log10(exp['z']), jnp.log10(exp['z_bins']))
    mu_start = jnp.linalg.solve(jnp.dot(interpol_matrix.T, interpol_matrix), jnp.dot(interpol_matrix.T, obs.mag))
    x0 = {'mu_bins': mu_start,
          'coef': jnp.array([3.1, 2.]),
          'variables': obs.variables.reshape(-1, obs.mag.shape[0]),
          'sigma_int': 0.15
    }
    
    L = lambda x: likelihood(x, exp, cov, obs, cosmo=binned_cosmo, truncated=False, restricted=False)
    
    params = dict(niter=1000,
        lmbda=1e4,
        tol=1e-2)
    params.update(kwargs)

    res, loss, lmbda, iter_params = tncg(
        L,
        x0,
        **params)

    return res, jax.hessian(L)(res), loss, lmbda, iter_params