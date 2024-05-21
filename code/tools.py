from scipy.linalg import block_diag
from edris.models import FullCovariance, Obs
import jax.numpy as jnp
import numpy as np
import edris
import pandas

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x


def dset_sanitize_and_filter(dset, return_index=True):
    logger.log(logging.INFO, "Cleaning skysurvey dataset")
    dset.data["detected"] = (dset.data["flux"] / dset.data["fluxerr"]) > 5
    dset.targets.data["keep"] = False
    dset.targets.data["good"] = False
    bands = np.unique(dset.data["band"])

    ids = np.unique(list(map(lambda x: x[0], dset.data.index)))
    for i in tqdm(ids):
        target = dset.targets.data.loc[i]
        obs_data = dset.data.loc[i]
        dset.targets.data.loc[i, "keep"] = np.any(obs_data["time"].between(target["t0"] - 10, target["t0"] + 25))
        
        dset.targets.data.loc[i, "good"] = (
            dset.targets.data.loc[i, "keep"]
            and np.any([np.sum(obs_data[obs_data["detected"] & (obs_data['band']==b)]["time"].between(target["t0"] - 40, target["t0"] + 130)) >= 10 for b in bands])
            and (np.sum(obs_data[obs_data["detected"]]["time"].between(target["t0"] - 40, target["t0"])) > 1)
            and (np.sum(obs_data[obs_data["detected"]]["time"].between(target["t0"], target["t0"] + 130)) > 1)
            # and len(np.unique(obs_data[obs_data["time"].between(target["t0"] - 40, target["t0"] + 130)]['band'])) > 2
        )
    logger.log(logging.INFO, "Done")
    if return_index:
        return np.where(dset.targets.data["good"])[0]




def X0X1C_to_MbX1C(values, cov, M0=10.501612):
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
    res, data, index, interest_vars=["x1", "c"], n_bins=10, M0=10.501612
):
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
    var = jnp.array([values[col].to_list() for col in interest_vars]).flatten()
    mag = jnp.array(values["Mb"].to_list())
    obs = Obs(mag, var)

    exp = {
        "z": jnp.array(data.loc[index]["z"].to_list()),
        "z_bins": edris.tools.log_bins(data.loc[index]["z"].min() - 1e-4, 0.06, n_bins),
    }
    logger.log(logging.INFO, "Done")
    return exp, cov, obs


def get_cov_edris(hess, invcov=False):
    n_var = len(hess["coef"]["coef"])
    n_bins = len(hess["mu_bins"]["mu_bins"])
    n = hess["variables"]["variables"].shape[1]
    row1 = jnp.hstack(
        (
            hess["coef"]["coef"],
            hess["coef"]["mu_bins"],
            hess["coef"]["variables"].reshape(n_var, n_var * n),
        )
    )
    row2 = jnp.hstack(
        (
            hess["mu_bins"]["coef"],
            hess["mu_bins"]["mu_bins"],
            hess["mu_bins"]["variables"].reshape(n_bins, n_var * n),
        )
    )
    row3 = jnp.hstack(
        (
            hess["variables"]["coef"].reshape(n_var * n, n_var),
            hess["variables"]["mu_bins"].reshape(n_var * n, n_bins),
            hess["variables"]["variables"].reshape(n_var * n, n_var * n),
        )
    )
    flatten_hessian = jnp.vstack((row1, row2, row3))
    if invcov:
        return 0.5 * flatten_hessian
    return jnp.linalg.inv(0.5 * flatten_hessian)
