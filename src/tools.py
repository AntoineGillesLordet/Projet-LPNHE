from scipy.linalg import block_diag
from edris.models import FullCovariance, Obs
import jax.numpy as jnp
import numpy as np
import edris
import pandas
from .logging import logger, logging

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x


def dset_sanitize_and_filter(dset, return_index=True):
    """
    Flags SN that are detected with the key ``keep`` and those that pass some cuts for lightcurve fitting with ``good``.
    
    Parameters
    ----------
    dset : skysurvey.DataSet
        DataSet of the SN containing both the targets data and lightcurves
    return_index : bool, optional
        If True, returns the index of the SN that will be fitted as a list, otherwise only the dset.targets.data dataframe will be updated
    """
    logger.log(logging.INFO, "Cleaning skysurvey dataset")
    
    dset.data["detected"] = (dset.data["flux"] / dset.data["fluxerr"]) > 5
    dset.targets.data["keep"] = False
    dset.targets.data["good"] = False
    
    bands = np.unique(dset.data["band"])

    ids = np.unique(list(map(lambda x: x[0], dset.data.index)))
    for i in tqdm(ids):
        target = dset.targets.data.loc[i]
        obs_data = dset.data.loc[i]
        # Flags SN that were not observed to correct the rate
        dset.targets.data.loc[i, "keep"] = np.any(obs_data["time"].between(target["t0"] - 10, target["t0"] + 25))
        
        dset.targets.data.loc[i, "good"] = (
            dset.targets.data.loc[i, "keep"] # SN should be observed
            and np.any([np.sum(obs_data[obs_data["detected"] & (obs_data['band']==b)]["time"].between(target["t0"] - 40, target["t0"] + 130)) >= 10 for b in bands]) # One band should have 10 data points
            and (np.sum(obs_data[obs_data["detected"]]["time"].between(target["t0"] - 40, target["t0"])) > 1) # At least one data point before t0
            and (np.sum(obs_data[obs_data["detected"]]["time"].between(target["t0"], target["t0"] + 130)) > 1) # At least one data point after t0
        )
    logger.log(logging.INFO, "Done")
    
    if return_index:
        return np.where(dset.targets.data["good"])[0]




def X0X1C_to_MbX1C(values, cov, M0=10.501612):
    """
    Transforms a (x0,x1,c) data set with their covariance matrix to a (Mb, x1, c) data set with the corresponding covariance matrix
    
    Parameters
    ----------
    values : pandas.DataFrame
        Values of (x0,x1,c) as a DataFrame.
    cov : dict
        Dict containing the covariance matrix of each individual event.
    M0 : float, optional
        Magnitude shift to apply when converting x0 to Mb

    Return
    ------
    new_values : pandas.DataFrame
        New DataFrame with the additional column ``Mb``
    new_covs : dict
        Dict of the new covariance matrices
    """
    logger.log(logging.INFO, "Converting (x0,x1,c) values to (Mb,x1,c)")
    new_values = values.copy()
    new_values["Mb"] = -2.5 * np.log10(values["x0"]) + M0
    new_covs = {
        i: jnp.matmul(
            jnp.matmul(
                jnp.array(
                    [
                        [-2.5 / (np.log(10) * values["x0"].loc[i]), 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ]
                ),
                cov[i],
            ),
            jnp.array(
                [
                    [-2.5 / (np.log(10) * values["x0"].loc[i]), 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
        )
        for i in values.index
    }
    logger.log(logging.INFO, "Done")
    return new_values, new_covs


def sncosmo_to_edris(
    res,
    data,
    index,
    n_bins=10,
    M0=10.501612
):
    """
    Transforms a skysurvey/sncosmo output to an edris input.
    
    Parameters
    ----------
    res : pandas.DataFrame
        Results of the sncosmo fit as returned by skysurvey.
    data : pandas.DataFrame
        Data of the SNe, should contain the redshifts of the SNe
    index : int list
        Index of the SN to use, should be included in the res index
    n_bins : int, optional
        Number of redshift bins to generate
    M0 : float, optional
        Magnitude shift to apply when converting x0 to Mb

    Return
    ------
    obs : edris.Obs
        Observations containing magnitudes and some standardisation variables
    cov : edris.CovMatrix
        Covariance matrix of the previous parameters (magnitudes and standardisation variables)
    exp : dict
        Explanatory variables. Contains the redshifts of the SN in ``z`` and the redshift bins in ``z_bins``.
    """
    logger.log(logging.INFO, "Creating input variables for edris")
    
    n = len(index)

    covariances = {
        i: np.array(res.loc[i].loc[["x0", "x1", "c"]][["cov_x0", "cov_x1", "cov_c"]])
        for i in index
    }

    stacked_res = pandas.DataFrame(
        {
            **{
                col: np.array(res["value"].loc[map(lambda x: (x, col), index)])
                for col in ["x0", "x1", "c"]
            }
        },
        index=index,
    )

    values, cov = X0X1C_to_MbX1C(stacked_res, covariances, M0=M0)

    full_matrix = block_diag(*[cov[i] for i in cov.keys()])
    full_cov_sorted = full_matrix[
        :,
        [
            *[3 * i for i in range(n)],
            *[3 * i + 1 for i in range(n)],
            *[3 * i + 2 for i in range(n)],
        ],
    ][
        [
            *[3 * i for i in range(n)],
            *[3 * i + 1 for i in range(n)],
            *[3 * i + 2 for i in range(n)],
        ],
        :,
    ]

    cov = FullCovariance(
        full_cov_sorted[:n, :n], full_cov_sorted[n:, n:], full_cov_sorted[:n, n:]
    )
    var = jnp.array([values[col].to_list() for col in ["x1", "c"]]).flatten()
    mag = jnp.array(values["Mb"].to_list())
    obs = Obs(mag, var)

    exp = {
        "z": jnp.array(data.loc[index]["z"].to_list()),
        "z_bins": edris.tools.log_bins(data.loc[index]["z"].min() - 1e-4, 0.06, n_bins),
    }
    logger.log(logging.INFO, "Done")
    return exp, cov, obs


def get_cov_from_hess(hess, invcov=False):
    """
    Obtains the covariance as a single matrix from the hessian evaluated at a point.
    
    Parameters
    ----------
    hess : dict dict
        Jax representation of the Hessian
    invcov : bool, optional
        If ``True`` returns the inverse of the covariance instead of the covariance.
    
    Return
    ------
    Cov : jax.numpy.array
        The covariance matrix as ``(0.5 H)^(-1)``
    """
    n_var = len(hess['coef']['coef'])
    n_bins = len(hess['mu_bins']['mu_bins'])
    n = hess['variables']['variables'].shape[1]
    row1 = jnp.hstack((hess['coef']['coef'], hess['coef']['mu_bins'], hess['coef']['variables'].reshape(n_var, n*n_var)))
    row2 = jnp.hstack((hess['mu_bins']['coef'], hess['mu_bins']['mu_bins'], hess['mu_bins']['variables'].reshape(n_bins,n*n_var)))
    row3 = jnp.hstack((hess['variables']['coef'].reshape(n*n_var,n_var),
                       hess['variables']['mu_bins'].reshape(n*n_var,n_bins),
                       hess['variables']['variables'].reshape(n*n_var,n*n_var)))
    flatten_hessian = jnp.vstack((row1, row2, row3))
    if invcov:
        return 0.5 * flatten_hessian
    return jnp.linalg.inv(0.5 * flatten_hessian)


def edris_filter(obs, cov, exp):
    """
    Filter edris explanatory variables, covariance and observation variables using standard cuts.
    SN kept have :
    * |c| < 0.3
    * err_c < 0.07
    * |x1| + err_x1 < 5
    
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
    obs : edris.Obs
        Filtered observations.
    cov : edris.CovMatrix
        Filtered covariance matrix
    exp : dict
        Filtered explanatory variables (only the ``z`` distribution is affected)

    """
    logger.log(logging.INFO, "Performing selection cut on color and stretch")
    n=len(obs.mag)
    
    goods = (jnp.sqrt(jnp.diag(cov.C_xx[n:,n:])) < 0.07) & (abs(obs.variables[n:]) < 0.3) & (jnp.sqrt(jnp.diag(cov.C_xx[:n,:n])) + jnp.abs(obs.variables[:n]) < 5)
    
    exp['z'] = exp['z'][goods]
    obs.mag = obs.mag[goods]
    obs.variables = obs.variables[jnp.tile(goods, 2)]
    cov_sel = cov.select(goods)
    
    return obs, cov, exp