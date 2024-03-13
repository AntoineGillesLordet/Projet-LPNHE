import skysurvey
import numpy as np
import pandas
import fitsio
from glob import glob
import pickle
from astropy.coordinates import SkyCoord
from astropy.units import deg,Mpc
from astropy import cosmology

try:
    from tqdm import tqdm
except:
    tqdm = lambda x:x

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

def load_ztf_sn_with_hosts(bgs_df):
    ztf_sn = pandas.read_csv('data/data_ztf.csv', index_col=0)
    
    if 'host' in ztf_sn.columns:
        return ztf_sn
    else:
        H0 = 67.66  # Hubble rate in km/s/Mpc
        Om0 = 0.3111  # Matter density parameter
        Ode0 = 1.0 - Om0  # Dark energy density parameter:
        cosmo = cosmology.LambdaCDM(H0, Om0, Ode0)

        SN_coords = SkyCoord(ra=ztf_sn["ra"]*deg, dec=ztf_sn["dec"]*deg, distance=cosmo.comoving_distance(ztf_sn['z']), unit=(deg,deg,Mpc))
        idx=[]

        ## Get galaxies that are close enough to be candidates
        for sn in tqdm(ztf_sn.index):
            ra, dec, z = ztf_sn.loc[sn]['ra'], ztf_sn.loc[sn]['dec'], ztf_sn.loc[sn]['z']
            idx.append(bgs_df[bgs_df['ra'].between(ra - 5, ra + 5) & bgs_df['dec'].between(dec - 5, dec + 5) & bgs_df['z'].between(z - 0.01, z+0.01)].index)
        
        nearest = []
        for i,bgs_ids in tqdm(zip(ztf_sn.index,idx), total=len(idx)):
            if len(bgs_ids):
                nearest.append(bgs_ids[SkyCoord(ra=bgs_df.loc[bgs_ids]['ra'],
                                               dec=bgs_df.loc[bgs_ids]['dec'],
                                               distance=cosmo.comoving_distance(bgs_df.loc[bgs_ids]['z']),
                                               unit=(deg,deg,Mpc)
                                              ).separation_3d(SN_coords[0]).argmin()])
            else:
                nearest.append(-1)

        ztf_sn['host'] = nearest
        ztf_sn['host_not_valid'] = np.array(nearest) < 0
        ztf_sn = ztf_sn.astype({"host": int})
        return ztf_sn