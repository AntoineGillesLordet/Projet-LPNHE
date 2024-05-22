import skysurvey
import numpy as np
import pandas
import fitsio
from glob import glob
import pickle
import healpy
from tqdm import tqdm
from .logging import logger
import logging


def load_bgs(
    path=None,
    filename="Uchuu.csv",
    columns=["RA", "DEC", "Z", "Z_COSMO", "STATUS"],
    in_desi=True,
):
    try:
        filepath = glob("../../**/" + filename, recursive=True)[0]
        df = pandas.read_csv(filepath, index_col=0)
        if (
            set(map(lambda s: s.lower(), columns))
            .difference({"status", "in_desi"})
            .issubset(set(df.columns))
        ):
            logger.log(logging.INFO, f"Found file {filepath} with columns {df.columns}")
            return df
        else:
            logger.log(logging.INFO,
                f"Found file {filepath} with columns {df.columns} but columns {columns} were prompted, defaulting to fits file"
            )
            raise IndexError
    except IndexError:
        logger.log(logging.INFO, "Reading BGS data from fits file")
        if path is None:
            path = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/Uchuu/LightCone/BGS_v2/BGS_LC_Uchuu.fits"

        data = fitsio.read(path, columns=columns)

        df = pandas.DataFrame(data)
        df.rename(columns=lambda s: s.lower(), inplace=True)
        for col in df.columns:
            if df[col].dtype == np.dtype(">f8") or df[col].dtype == np.dtype(">f4"):
                df[col] = np.float64(df[col])
            elif df[col].dtype == np.dtype(">i4"):
                df[col] = np.int64(df[col])

        if in_desi and "status" in df.columns:
            df["in_desi"] = df["status"] & 2**1 != 0
            df.drop(columns=["status"], inplace=True)
    logger.log(logging.INFO, "Done")
    return df[df["in_desi"]]


def extract_ztf(start_time=58179, end_time=59215):
    logger.log(logging.INFO, "Loading ZTF survey")
    with open("data/ztf_survey.pkl", "rb") as file:
        survey = pickle.load(file)
    survey.set_data(
        survey.data[(survey.data["mjd"] > start_time) & (survey.data["mjd"] < end_time)]
    )
    return survey


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
        map_ = healpy.smoothing(map_, fwhm=G)
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
