import skysurvey
import numpy as np
import pandas
import fitsio
from glob import glob
import pickle


def load_bgs(path=None, filename='Uchuu.csv', columns=['RA', 'DEC', 'Z', 'Z_COSMO', 'STATUS'], in_desi=True):
    try:
        filepath = glob('../../**/' + filename, recursive=True)[0]
        df = pandas.read_csv(filepath, index_col=0)
        if set(map(lambda s: s.lower(), columns)).difference({'status','in_desi'}).issubset(set(df.columns)):
            print(f'Found file {filepath} with columns {df.columns}')
            return df
        else:
            print(f'Found file {filepath} with columns {df.columns} but columns {columns} were prompted, defaulting to fits file')
            raise IndexError
    except IndexError:

        if path is None:
            path = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/Uchuu/LightCone/BGS_v2/BGS_LC_Uchuu.fits"

        data = fitsio.read(path, columns=columns)

        df = pandas.DataFrame(data)
        df.rename(columns=lambda s: s.lower(), inplace=True)
        for col in df.columns:
            if (df[col].dtype == np.dtype('>f8') or df[col].dtype == np.dtype('>f4')):
                df[col] = np.float64(df[col])
            elif df[col].dtype == np.dtype('>i4'):
                df[col] = np.int64(df[col])

        if in_desi and 'status' in df.columns:
            df['in_desi'] = df['status'] & 2**1 != 0
            df.drop(columns=['status'], inplace=True)

    return df[df['in_desi']]

def extract_ztf(start_time=58179, end_time=59215):
    with open('data/ztf_survey.pkl','rb') as file:
        survey = pickle.load(file)
    survey.set_data(survey.data[(survey.data['mjd'] > start_time) & (survey.data['mjd'] < end_time)])
    return survey
