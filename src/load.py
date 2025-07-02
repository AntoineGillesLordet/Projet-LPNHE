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
import os

from nacl.dataset import TrainingDataset
from nacl.models.salt2 import SALT2Like

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
            filename=filename
        elif not os.path.exists(filename):
            filename = glob("../../**/" + filename, recursive=True)[0]
        df = pandas.read_csv(filename, index_col=0)
        if (
            set(map(lambda s: s.lower(), columns))
            .difference({"status", "in_desi"})
            .issubset(set(df.columns))
        ):
            logger.log(logging.INFO, f"Found .csv file at {filename} with columns {df.columns}")
            return df
        else:
            logger.log(
                logging.INFO,
                f"Found file {filename} at with columns {df.columns} but columns {columns} were prompted, loading from the fits file",
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


def extract_ztf(path='data/ztf_survey.pkl', start_time=58179, end_time=59215):
    """
    Load the ztf logs from pkl file and restrict it to the DR2.5.
    """
    logger.log(logging.INFO, "Loading ZTF survey")
    with open(path, "rb") as file:
        ztf = pickle.load(file)
    if start_time and end_time:
        ztf.set_data(
            ztf.data[(ztf.data["mjd"] > start_time) & (ztf.data["mjd"] < end_time)]
        )
    ztf.data.band = ztf.data.band.replace({'ztfi':'ztf::i', 'ztfr':'ztf::r', 'ztfg':'ztf::g'})
    return ztf

def extract_snls(path='data/snls_obslogs_cured.csv'):
    """
    Load the snls logs and create the survey
    """
    from shapely import geometry
    snls = skysurvey.GridSurvey.from_pointings(data=pandas.read_csv(path, encoding='utf-8'),
                                footprint=geometry.box(-0.5, -0.5, 0.5, 0.5),
                                fields_or_coords={'D1': {'ra': 36.450190, 'dec': -4.45065},
                                        'D2': {'ra': 150.11322, 'dec': +2.21571},
                                        'D3': {'ra': 214.90738, 'dec': +52.6660},
                                        'D4': {'ra': 333.89903, 'dec': -17.71961}},)

    snls.data.band = snls.data.band.apply(lambda x : 'MEGACAM6::' + x.split(':')[-1])
    snls.data.replace({'MEGACAM6::y':'MEGACAM6::i2'}, inplace=True)
    return snls

def extract_hsc(path='data/hsc_logs_realistic_skynoise.csv'):
    """
    Load the hsc logs and create the survey
    """
    from shapely import geometry
    hsc = skysurvey.Survey.from_pointings(pandas.read_csv(path, index_col=0), 
                                          geometry.Point(0,0).buffer(0.7))
    return hsc

def load_from_skysurvey(path, survey=None):
    """
    Loads a pickle file containing the sn and lc tables from a skysurvey dataset.
    If a survey is provided, adds the corresponding SNe names to the tables and index the sn table by name.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
        lc = pickle.load(f)
    if survey:
        data["survey"] = survey
        lc["survey"] = survey
        data["sn"] = data.survey + '_' + data.index.astype(str)
        lc["sn"] = lc.survey + '_' + lc.index.get_level_values(0).astype(str)
        data.set_index(data["sn"], inplace=True)
        lc.set_index(lc["sn"].values, inplace=True)


    float32_cols = list(data.select_dtypes(include='float32'))
    data[float32_cols] = data[float32_cols].astype('float64')
    
    float32_cols = list(lc.select_dtypes(include='float32'))
    lc[float32_cols] = lc[float32_cols].astype('float64')

    return data, lc

def load_from_pets(path, survey=None):
    """
    Loads the pets csv files containing sn and lc tables.
    If a survey is provided, adds the corresponding SNe names to the tables and index the sn table by name.
    """

    lc = pandas.read_csv(path+"/mock_lc.csv", index_col=0)
    data = pandas.read_csv(path+"/mock_sne.csv", index_col=0)
    
    if survey:
        data["survey"] = survey
        lc["survey"] = survey
        data["name"] = data.survey + '_' + data.sn.astype(str)
        lc["name"] = lc.survey + '_' + lc.sn.astype(str)

        data.set_index(data["name"], inplace=True)
        lc.set_index(lc["name"].values, inplace=True)
        
    lc.valid = lc.valid.astype(bool)
    data.valid = data.valid.astype(bool)
    data.valid*=data.name.isin(lc.name)
    data.rename(columns={'zhel':'z'}, inplace=True)
    return data, lc