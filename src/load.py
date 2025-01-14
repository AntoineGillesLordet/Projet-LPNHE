import skysurvey
import numpy as np
import pandas
import fitsio
from glob import glob
import pickle
import healpy
from tqdm.auto import tqdm
from .logging import logger
import logging

from nacl.dataset import TrainingDataset
from nacl.models.salt2 import SALT2Like
from nacl.specutils import clean_and_project_spectra

from lemaitre import bandpasses
filterlib = bandpasses.get_filterlib()

def load_bgs(
    path=None,
    filename="Uchuu.csv",
    columns=["RA", "DEC", "Z", "Z_COSMO", "STATUS"],
    in_desi=True,
):
    try:
        if filename[0] == "/":
            filepath=filename
        else:
            filepath = glob("../../**/" + filename, recursive=True)[0]
        df = pandas.read_csv(filepath, index_col=0)
        if (
            set(map(lambda s: s.lower(), columns))
            .difference({"status", "in_desi"})
            .issubset(set(df.columns))
        ):
            logger.log(logging.INFO, f"Found .csv file at {filepath} with columns {df.columns}")
            return df
        else:
            logger.log(
                logging.INFO,
                f"Found file {filepath} at with columns {df.columns} but columns {columns} were prompted, loading from the fits file",
            )
    except IndexError:
        logger.log(logging.WARNING, f"No file named {filename} around here, trying to load from fits file")
    
    if path is None:
        path = "/pscratch/sd/a/agillesl/FirstGenMocks/Uchuu/LightCone/BGS_v2/BGS_LC_Uchuu.fits"
    logger.log(logging.INFO, f"Reading BGS data from fits file at {path}")

    data = fitsio.read(path, columns=columns)

    df = pandas.DataFrame(data)
    df.rename(columns=lambda s: s.lower(), inplace=True)
    for col in df.columns:
        if df[col].dtype == np.dtype(">f8") or df[col].dtype == np.dtype(">f4"):
            df[col] = np.float64(df[col])
        elif df[col].dtype == np.dtype(">i4") or df[col].dtype == np.dtype(">i8"):
            df[col] = np.int64(df[col])

    if in_desi:
        if "status" in df.columns:
            df["in_desi"] = df["status"] & 2**1 != 0
            df = df[df["in_desi"]]
            df.drop(columns=["status"], inplace=True)
        else:
            logging.log(logging.WARNING, "Only galaxies in DESI footprint requested but the dataset doesn't contain the 'status' column")

    logger.log(logging.INFO, "Done")
    return df


def extract_ztf(start_time=58179, end_time=59215):
    logger.log(logging.INFO, "Loading ZTF survey")
    with open("data/ztf_survey.pkl", "rb") as file:
        survey = pickle.load(file)
    survey.set_data(
        survey.data[(survey.data["mjd"] > start_time) & (survey.data["mjd"] < end_time)]
    )
    return survey

def extract_snls():
    from shapely import geometry
    snls = skysurvey.GridSurvey.from_pointings(data=pandas.read_csv('data/snls_obslogs_cured.csv', encoding='utf-8'),
                                footprint=geometry.box(-0.5, -0.5, 0.5, 0.5),
                                fields_or_coords={'D1': {'ra': 36.450190, 'dec': -4.45065},
                                        'D2': {'ra': 150.11322, 'dec': +2.21571},
                                        'D3': {'ra': 214.90738, 'dec': +52.6660},
                                        'D4': {'ra': 333.89903, 'dec': -17.71961}},)

    snls.data.band = snls.data.band.apply(lambda x : 'megacam6::' + x[-1])
    snls.data.replace({'megacam6::i':'megacam6::i2', 'megacam6::y':'megacam6::i2'}, inplace=True)
    return snls


def load_maps(
    ztf_path="data/ztf_distribution.fits",
    bgs_pix_path="data/bgs_pixels.pkl",
    ztf_nside=128,
    bgs_nside=128,
    F=0.1,
):
    logger.log(logging.INFO, "Loading ZTF skymap and BGS redshift distribution")
    try:
        map_ = healpy.read_map(ztf_path)

        with open(bgs_pix_path, "rb") as fp:
            bgs_redshifts = pickle.load(fp)

    except FileNotFoundError:
        logger.log(logging.INFO, "Maps file missing, computing them again and saving")

        # Initial SN density skymap
        ztf_sn = pandas.read_csv("data/data_ztf.csv", index_col=0)
        ids = healpy.ang2pix(
            theta=np.pi / 2 - ztf_sn["dec"] * np.pi / 180,
            phi=ztf_sn["ra"] * np.pi / 180,
            nside=ztf_nside,
        )
        map_ = np.zeros(healpy.nside2npix(ztf_nside))
        for i in tqdm(ids):
            map_[i] += 1
        map_ = healpy.smoothing(map_, fwhm=F)
        map_ -= map_.min()

        bgs_df = load_bgs(
            columns=[
                "RA",
                "DEC",
                "Z",
                "Z_COSMO",
                "STATUS",
                "V_PEAK",
                "V_RMS",
                "R_MAG_ABS",
                "R_MAG_APP",
            ]
        )

        # Mask of areas where there are no BGS galaxies
        id_bgs = healpy.ang2pix(
            theta=np.pi / 2 - bgs_df["dec"] * np.pi / 180,
            phi=bgs_df["ra"] * np.pi / 180,
            nside=nside,
        )
        mask = np.zeros(healpy.nside2npix(nside), dtype=bool)
        for i in tqdm(id_bgs):
            mask[i] = True
        map_[~mask] = 0

        # Normalization and saving
        map_ /= np.sum(map_)
        healpy.write_map("data/ztf_distribution.fits", map_)

        # list of BGS redshifts along lines of sight
        z_nside = 64
        id_bgs = healpy.ang2pix(
            theta=np.pi / 2 - bgs_df["dec"] * np.pi / 180,
            phi=bgs_df["ra"] * np.pi / 180,
            nside=bgs_nside,
        )
        bgs_pix = [[]] * healpy.nside2npix(bgs_nside)
        for i, nb_pix in tqdm(enumerate(id_bgs), total=len(id_bgs)):
            bgs_pix[nb_pix] = bgs_pix[nb_pix] + [id_bgs.index[i]]

        bgs_redshifts = [np.array(bgs_df.loc[ids]["z"]) for ids in bgs_pix]

        with open("data/bgs_redshifts_map.pkl", "wb") as fp:
            pickle.dump(bgs_redshifts, fp)
        logger.log(logging.INFO, "Done")

    finally:
        return map_, bgs_redshifts

def make_tds_from_pets(sne_data, lc_data, sp_data, sigma_x1_lim=0.2, sigam_c_lim=0.02):
    sne_data.valid=sne_data.valid.astype(bool)
    sne_data["valid"] = sne_data["valid"] & (sne_data["err_x1"] < sigma_x1_lim) & (sne_data["err_c"] < sigam_c_lim)
    sne_data.rename(columns={'t0':'tmax', 'zhel':'z'}, inplace=True)

    lc_data.set_index([lc_data.sn, lc_data.index], inplace=True)
    lc_data.index.names= [None, None]
    lc_data['x'] = 0.
    lc_data['y'] = 0.
    lc_data['sensor_id'] = 0
    lc_data.rename(columns={"time":"mjd"}, inplace=True)
    lc_data.index.names = [None, None]
    lc_data.valid=lc_data.valid.astype(bool)
    
    sp_data['i_basis'] = 0
    sp_data.valid=sp_data.valid.astype(bool)
    sp_data.rename(columns={"snid":"sn", "time":"mjd", 'flux_true':'fluxtrue'}, inplace=True)

    # Clean points corresponding to invalid SN
    for sn in tqdm(sne_data.sn[~sne_data.valid], desc="Clearing LC and spectra according to PeTs"):
        sp_data.loc[sp_data['sn']==sn, "valid"] = False
        lc_data.loc[lc_data['sn']==sn, "valid"] = False

    tds = TrainingDataset(sne=sne_data.to_records(),
                          lc_data=lc_data.to_records(),
                          spec_data=sp_data.to_records(),
                          filterlib=filterlib)
    SALT2Like.flag_out_of_range_datapoints(tds, wl_range=(2000., 11000.), basis_knots=[200, 20], compress=True)
    model = SALT2Like(tds)
    # Project spectra onto spline basis
    projected_spectra, in_error = clean_and_project_spectra(tds, model.basis.bx)
    
    return TrainingDataset(tds.sn_data.nt, lc_data=tds.lc_data.nt,
                           spec_data=np.rec.array(np.hstack(projected_spectra)),
                           basis=model.basis.bx,
                           filterlib=filterlib)