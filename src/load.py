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


def extract_ztf(path='data/ztf_survey.pkl', start_time=58179, end_time=59215):
    """
    Load the ztf logs from pkl file and restrict it to the DR2.5.
    """
    logger.log(logging.INFO, "Loading ZTF survey")
    with open(path, "rb") as file:
        survey = pickle.load(file)
    survey.set_data(
        survey.data[(survey.data["mjd"] > start_time) & (survey.data["mjd"] < end_time)]
    )
    return survey

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

    snls.data.band = snls.data.band.apply(lambda x : 'megacam6::' + x[-1])
    snls.data.replace({'megacam6::i':'megacam6::i2', 'megacam6::y':'megacam6::i2'}, inplace=True)
    return snls

def extract_hsc(path='data/hsc_logs_realistic_skynoise.csv'):
    """
    Load the hsc logs and create the survey
    """
    from shapely import geometry
    survey = skysurvey.Survey.from_pointings(pandas.read_csv(path, index_col=0), 
                                             geometry.Point(0,0).buffer(0.7))


def make_tds_from_pets(sne_data, lc_data, sp_data, sigma_x1_lim=0.2, sigam_c_lim=0.02):
    """
    Construct a NaCl training dataset from pets output sn and lc tables.
    Additionally apply cuts on the x1 and c errors.
    """
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

def load_from_skysurvey(path, survey=None):
    """
    Loads a pickle file containing the sn and lc tables from a skysurvey dataset.
    If a survey is provided, adds the corresponding SNe names to the tables and index the sn table by name.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
        lc = pickle.load(f)
    data["sn"] = data.index
    lc["sn"] = lc.index.get_level_values(0)
    if survey:
        data["survey"] = survey
        lc["survey"] = survey
        data["name"] = data.survey + '_' + data.sn.astype(str)
        lc["name"] = lc.survey + '_' + lc.sn.astype(str)
        data.set_index(data["name"], inplace=True)
        lc.set_index(lc["name"].values, inplace=True)


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