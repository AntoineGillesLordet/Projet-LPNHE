import skysurvey
import numpy as np
import pandas
import fitsio
from glob import glob


def load_bgs(path=None, filename='Uchuu.csv', columns=['RA', 'DEC', 'Z', 'Z_COSMO', 'STATUS']):
    try:
        filepath = glob('../../**/' + filename, recursive=True)[0]
        df = pandas.read_csv(filepath, index_col=0)
        print(f'Found file {filepath} with columns {df.columns}')
    except IndexError:

        if path is None:
            path = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/Uchuu/LightCone/BGS_v2/BGS_LC_Uchuu.fits"

        data = fitsio.read(path, columns=columns)

        df = pandas.DataFrame(data)
        df.rename(columns={'RA':'ra', 'DEC':'dec','Z':'z', 'Z_COSMO':'z_cosmo'}, inplace=True)
        for col in df.columns:
            if (df[col].dtype == np.dtype('>f8') or df[col].dtype == np.dtype('>f4')):
                df[col] = np.float64(df[col])
            elif df[col].dtype == np.dtype('>i4'):
                df[col] = np.int64(df[col])

        if 'STATUS' in df.columns:
            df['in_desi'] = df['STATUS'] & 2**1 != 0
            df.drop(columns=['STATUS'], inplace=True)

    return df
