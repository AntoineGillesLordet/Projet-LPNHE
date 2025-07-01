from scipy.linalg import block_diag
from edris.models import FullCovariance, Obs
import jax.numpy as jnp
import numpy as np
import pandas
import healpy

from astropy.cosmology import Planck18, Planck15
from edris.tools import log_bins
from .logging import logger, logging

try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x: x


def dset_sanitize_and_filter(dset, return_index=True, all_pts=False):
    """
    Flags SN that are detected with the key ``keep`` and those that pass some cuts for lightcurve fitting with ``valid``.

    Parameters
    ----------
    dset : skysurvey.DataSet
        DataSet of the SN containing both the targets data and lightcurves
    return_index : bool, optional
        If True, returns the index of the SN that will be fitted as a list, otherwise only the dset.targets.data dataframe will be updated
    """
    logger.log(logging.INFO, "Correcting rate to observations and filtering based on observations")

    if all_pts:
        dset.data["detected"] = True
    else:
        dset.data["detected"] = (dset.data["flux"] / dset.data["fluxerr"]) > 5

    dset.targets.data["keep"] = False
    dset.targets.data["valid"] = False

    bands = np.unique(dset.data["band"])

    ids = np.unique(list(map(lambda x: x[0], dset.data.index)))
    for i in tqdm(ids):
        target = dset.targets.data.loc[i]
        obs_data = dset.data.loc[i]
        # Flags SN that were not observed to correct the rate
        dset.targets.data.loc[i, "keep"] = np.any(
            obs_data["time"].between(target["t0"] - 10, target["t0"] + 15)
        )

        dset.targets.data.loc[i, "valid"] = (
            dset.targets.data.loc[i, "keep"]  # SN should be observed
            and np.sum(
                [
                    np.sum(
                        obs_data[obs_data["detected"] & (obs_data["band"] == b)][
                            "time"
                        ].between(target["t0"] - 30, target["t0"] + 100)
                    )
                    >= 5
                    for b in bands
                ]
            ) >= 2 # Two bands should have 5 or more data points
            and (
                np.sum(
                    obs_data[obs_data["detected"]]["time"].between(
                        target["t0"] - 30, target["t0"]
                    )
                )
                > 1
            )  # At least one data point before t0
            and (
                np.sum(
                    obs_data[obs_data["detected"]]["time"].between(
                        target["t0"], target["t0"] + 100
                    )
                )
                > 1
            )  # At least one data point after t0
        )
    logger.log(logging.INFO, "Done")

    if return_index:
        return dset.targets.data[dset.targets.data["valid"]].index


def X0X1C_to_MbX1C(values, cov, M0=10.501612, alpha=0.14, beta=3.15):
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


def sncosmo_to_edris(res, data, index, n_bins=10, M0=10.501612):
    """
    Transforms a skysurvey/sncosmo output to an edris input, and filters SN using standard cuts.
    
    SN kept have :
    * abs(c) < 0.3
    * err_c < 0.07
    * abs(x1) + err_x1 < 5


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

    goods = (
        (jnp.sqrt(jnp.diag(cov.C_xx[n:, n:])) < 0.07)
        & (abs(obs.variables[n:]) < 0.3)
        & (jnp.sqrt(jnp.diag(cov.C_xx[:n, :n])) + jnp.abs(obs.variables[:n]) < 5)
    )
    obs.mag = obs.mag[goods]
    obs.variables = obs.variables[jnp.tile(goods, 2)]
    cov_sel = cov.select(goods)
    
    data['used_edris'] = False
    data.loc[index[goods], 'used_edris'] = True
    
    exp = {
        "z": jnp.array(data[data['used_edris']]["z"].to_list()),
        "z_bins": log_bins(data[data['used_edris']]["z"].min() - 1e-4, data[data['used_edris']]["z"].max() + 1e-2, n_bins),
    }
    logger.log(logging.INFO, "Done")
    return exp, cov_sel, obs


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
    n = {k1:hess[k1][k1].shape[1] for k1 in hess.keys()}
    n["variables"]*=n["coef"]
    flatten_hess = jnp.vstack([np.hstack([hess[k1][k2].reshape(n[k1], n[k2]) for k2 in hess.keys()]) for k1 in hess.keys()])
    if invcov:
        return 0.5 * flatten_hessian
    return jnp.linalg.inv(0.5 * flatten_hessian)


def edris_filter(exp, cov, obs, data):
    """

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
    n = len(obs.mag)

    goods = (
        (jnp.sqrt(jnp.diag(cov.C_xx[n:, n:])) < 0.07)
        & (abs(obs.variables[n:]) < 0.3)
        & (jnp.sqrt(jnp.diag(cov.C_xx[:n, :n])) + jnp.abs(obs.variables[:n]) < 5)
    )

    exp["z"] = exp["z"][goods]
    obs.mag = obs.mag[goods]
    obs.variables = obs.variables[jnp.tile(goods, 2)]
    cov_sel = cov.select(goods)
    
    data['used_edris'] = False
    data.loc[np.where(goods)[0], 'used_edris'] = True
    
    return exp, cov_sel, obs


def mag_Planck18(x):
    return jnp.array(Planck18.distmod(np.array(x))) - 19.3


def mag_Planck15(x):
    return jnp.array(Planck15.distmod(np.array(x))) - 19.3

def wrapp_around(ra, dec, unit_in='degree', unit_out='rad'):
    if unit_in=='rad':
        ra*=180/np.pi
        dec*=180/np.pi
    elif unit_in!='degree':
        raise KeyError(f"Incoming unit should be either 'rad' or 'degree' but {unit_in} was provided")
    flip = dec > 90
    dec[flip] = 90 - dec[flip]
    ra[flip] += 180

    flip = dec < -90
    dec[flip] = -90 - dec[flip]
    ra[flip] -= 180
    ra = np.mod(ra + 180, 360) - 180
    if unit_out=='degree':
        return ra, dec
    elif unit_out=='rad':
        return ra*np.pi/180, dec*np.pi/180
    else:
        raise KeyError(f"Returned unit should be either 'rad' or 'degree' but {unit_out} was provided")


def halo_den_profile(r, rho, R, alpha=0.16):
    """
    Einasto profile for halo density
    """
    return rho*np.exp(-2/alpha*(np.power(r/R, alpha) - 1))


def gen_mask(nside):
    """
    Quick generation of the ztf mask for healpy maps
    
    """
    import ztffields
    good_fields = ztffields.get_fieldid(galb_range=[[-90,-10],[10,90]])

    
    ra_pix, dec_pix = healpy.pix2ang(nside, np.arange(healpy.nside2npix(nside)), lonlat=True)
    radecs_pix = pandas.DataFrame({"ra":ra_pix, "dec":dec_pix})
    hp_pix_to_ztf_field = ztffields.radec_to_fieldid(radecs_pix)
    filt_pix = hp_pix_to_ztf_field[~hp_pix_to_ztf_field.isin(good_fields).groupby('index_radec').any()].index
    mask = np.zeros(healpy.nside2npix(nside))
    mask[filt_pix] = 1
    mask[dec_pix < -30] = 1
    return mask


def create_map(halos, nside, map=None, col=None,):
    """
    Create a healpy map from a halo catalog
    """
    
    return_map = map is None
    if return_map:
        map = np.zeros(healpy.nside2npix(nside))
    index = healpy.ang2pix(nside, halos.ra, halos.dec, lonlat=True)
    if col=='mass':
        pix_value = pandas.DataFrame({'pix_id':index, 'h_mass':halos.BoundM200Crit.values}).groupby('pix_id').sum()
    elif col=='vpec':
        pix_value = pandas.DataFrame({'pix_id':index, 'h_vpec':halos.vpec.values}).groupby('pix_id').mean()
    else:
        pix_value = pandas.DataFrame({'pix_id':index, 'h_mass':np.ones(index.shape)}).groupby('pix_id').count()
    for pix_id in pix_value.index:
        map[pix_id] += pix_value.loc[pix_id].values[0]
    if return_map:
        return map