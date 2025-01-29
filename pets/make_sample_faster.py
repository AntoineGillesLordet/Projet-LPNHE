import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sncosmo
from astropy.table import Table,vstack
from lemaitre import bandpasses
from joblib import Parallel, delayed
import pickle
from astropy.coordinates import SkyCoord
from dustmaps.config import config
config.reset()

import warnings
from iminuit.warnings import IMinuitWarning

warnings.filterwarnings("ignore", category=IMinuitWarning)
from tqdm.auto import tqdm

#Constant
c_light=299792.452
#params of CMB dipole w.r.t. heliocentric frame (Planck Collaboration 2020)
CMB_V = 369.82 
CMB_l = 264.021 # galactic longitude
CMB_b = 48.253 # galactic latitude

#Function
def zcmb(z_hel, RA, DEC):
    '''heliocentric to CMB redshift conversion.
    z_hel (float):  heliocentric redshift of object
    RA (str): (degrees)  J2000 Right Ascension
    DEC (str): (degrees) J2000 declination
    '''
    RA_dd, DEC_dd	= float(RA), float(DEC)

    v_hel = z_hel*c_light
    cmb_dir = SkyCoord(l=CMB_l, b=CMB_b, frame='galactic', unit='deg')
    obj_dir = SkyCoord(RA_dd, DEC_dd, frame='icrs', unit='deg')
    ang_sep = cmb_dir.separation(obj_dir).value*np.pi/180
    v_corr = CMB_V*np.cos(ang_sep)
    vcmb = v_hel + v_corr
    return vcmb/c_light

def zcmb_tot(z_hel, RA, DEC):
    tt=np.vectorize(zcmb)
    return tt(z_hel, RA, DEC)

def get_mwebv(ra, dec, dustmap="planck"):
    """ get the mikly way E(B-V) extinction parameter for input coordinates

    This is based on dustmaps. 
    If this is the first time you use it, you may have to download the maps 
    first (instruction will be given)

    ra, dec: float, array
    coordinates

    which: string
    name of the dustmap to use.
    - planck: Planck 2013
    - sfd:
    - sf11:  A scaling of 0.86 is applied to the SFD map values to reflect the recalibration by Schlafly & Finkbeiner (2011)
    """
    coords = SkyCoord(ra, dec, unit="deg")
    if dustmap.lower() == "planck":
        from dustmaps.planck import PlanckQuery as dustquery

        ebv=dustquery()(coords)
    elif (dustmap.lower() == "sfd") :
        from dustmaps.sfd import SFDQuery as dustquery
        ebv=dustquery()(coords)
    elif (dustmap.lower() == "sf11"):
        from dustmaps.sfd import SFDQuery as dustquery
        ebv=dustquery()(coords)*0.86

    else:
        raise NotImplementedError("Only Planck, SFD, and SF11 maps implemented")

    return ebv


modelpath='../data/SALT_snf/'
m0file='nacl_m0_test.dat'
m1file='nacl_m1_test.dat'
clfile='nacl_color_law_test.dat'



with open('/cfs/data/angi0819/Projet_LPNHE/dataset_hsc.pkl', 'rb') as f:
    # dset = pickle.load(f)
    data = pickle.load(f)
    lc = pickle.load(f)
# data = dset.targets.data
# lc = dset.data

def create_ztf_lc_tds(data,lc):
    lc_tot=lc.copy()
    lc_tot.reset_index('index', names='sn', inplace=True)
    lc_tot.reset_index(drop=True, inplace=True)
    
    # Select only points at 5 sigma and in [tmax-50, tmax+100]
    lc_tot = lc_tot[(lc_tot.flux/lc_tot.fluxerr>5) &
                    (lc_tot.time.between(data.loc[lc_tot.sn, 't0'].reset_index(drop=True) - 50,
                                         data.loc[lc_tot.sn, 't0'].reset_index(drop=True) + 100))].copy()
    lc_tot.reset_index(drop=True, inplace=True)
    
    # Select SN on : >=5 detections in >=2 bands
    goods_sn = (lc_tot.groupby(["sn"]).band.nunique() >= 2) & (lc_tot.groupby(["sn"])['flux'].count() >= 5)
    
    # Drop all the photometry points of SN that did not satisfy the previous condition
    lc_tot = lc_tot[goods_sn.loc[lc_tot.sn].reset_index(drop=True)]

    # Quick LC indexing : grouping by sn and band, count() reduces the dataframe to a single line for a given sn and band
    grouped = lc_tot.groupby(["sn", "band"]).count()
    # Mapping between the tuple (sn, band) and the lc index
    lc_ids = {idx:i for i, idx in enumerate(grouped.index)}
    # Apply the mapping to all the photometry points
    lc_tot["lc"] = lc_tot.apply(lambda x : lc_ids[(x.sn, x.band)], axis=1)
    
    # Filling the other columns
    lc_tot['name'] = lc_tot.sn
    
    lc_tot.rename(columns={"time":"mjd"}, inplace=True)
    lc_tot.insert (5, 'magsys', 'AB')
    lc_tot.insert (6, 'exptime', 'NaN')
    lc_tot.insert (7, 'valid', 1)
    lc_tot.insert (10, 'mag_sky', 'NaN')
    lc_tot.insert (11, 'seeing', 'NaN')
    
    # Sorting by SN number because it's nicer
    lc_tot = lc_tot.sort_values('sn').reset_index(drop=True)
    
    float32_cols = list(lc_tot.select_dtypes(include='float32'))
    lc_tot[float32_cols] = lc_tot[float32_cols].astype('float64')

    lc_tot.to_csv('mock_lcs.csv', index=False)
    return lc_tot




### DATA file


def get_SN_data(data):	
    data=data[['z','ra','dec','t0','x0','x1','c']].copy()

    data['name']=data.index.values
    data['sn']=data.index.values
    data['classification']='SNIa'
    data['comments']=''

    data=data.rename(columns={'z': 'zhel', 't0': 'tmax'})              

    data['mwebv']=get_mwebv(data['ra'].values, data['dec'].values, dustmap='planck')

    data['zcmb']=zcmb_tot(data['zhel'],data['ra'], data['dec'])
    float32_cols = list(data.select_dtypes(include='float32'))
    data[float32_cols] = data[float32_cols].astype('float64')


    data.to_csv('mock_sne.csv', index=False)
    return data

def remove_outlier_fit(name, lc, sne, sigma=3):
    import logging
    logging.basicConfig(level=logging.WARNING)
    from iminuit.warnings import IMinuitWarning
    warnings.filterwarnings("ignore", category=IMinuitWarning)
    from lemaitre import bandpasses
    filterlib=bandpasses.get_filterlib()
    
    
    #Select data for specific SN
    lc_sn=lc[lc.name==name]


    #Redshift and MW extinction
    mwebv=sne[sne.name==name].mwebv.values[0]
    zsn=sne[sne.name==name].zhel.values[0]
    #Transform to Table for sncosmo
    lc_sncosmo=Table.from_pandas(lc_sn[['sn','name','mjd','flux','fluxerr','magsys','exptime','valid','lc','band','mag_sky','seeing','zp']])

    # create a model
    source = sncosmo.SALT2Source(modeldir=modelpath,m0file=m0file, m1file=m1file, clfile=clfile)
    dust = sncosmo.CCM89Dust()
    model= sncosmo.Model(source=source, effects=[dust],effect_names=['mw'],effect_frames=['obs'])

    model.set(z=zsn, mwebv=mwebv, mwr_v=3.1)  # set the model's redshift and MW
    #First fit to get a starting position for MCMC
    res, mod = sncosmo.fit_lc(lc_sncosmo, model,['t0', 'x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)

    residu=0
    counter=0
    while np.size(residu)>0:
        counter += 1
        if counter == 5:
            break
        residu=[]
        for i,j in enumerate(np.unique(lc_sncosmo[lc_sncosmo['valid']==1]['band'])):

            lc_band=lc_sncosmo[(lc_sncosmo['band']==j) &(lc_sncosmo['valid']==1)]

            residual=lc_band['flux']-mod.bandflux(j, lc_band['mjd'], zp=lc_band['zp'], zpsys=lc_band['magsys'])
            #3sigma clipping
            if (len(residual) > 0) and (np.std(residual) > 0):
                idx=np.where(np.abs(residual-np.mean(residual))/np.std(residual)>sigma)[0]

                pos_band=min(np.where(lc_sncosmo[(lc_sncosmo['valid']==1)]['band']==j)[0])
                idx=idx+pos_band

                residu.append(idx)

        residu=np.concatenate(residu)
        lc_sncosmo['valid'][residu]=0

        res, mod = sncosmo.fit_lc(lc_sncosmo[lc_sncosmo['valid']==1], model,['t0', 'x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)
    return lc_sncosmo



def make_full_sample(data,lcdata):
    #Create ZTf sne training dataset  
    sne = get_SN_data(data)
    #Create ZTf spectra training dataset  
    lc = create_ztf_lc_tds(data,lcdata)

    lc_tot = Table()
    
    with Parallel(n_jobs=64) as parallel:
        #Remove photometric outliers:
        lc_outs = parallel(delayed(remove_outlier_fit)(sn, lc, sne, sigma=3) for i,sn in tqdm(enumerate(np.unique(lc.name)), desc="Removing outlier"))
        lc_tot=vstack(lc_outs)
        lc_tot=pd.DataFrame(np.array(lc_tot),columns=['sn', 'name', 'mjd', 'flux', 'fluxerr', 'magsys', 'exptime', 'valid','lc', 'band', 'mag_sky', 'seeing', 'zp'])
        lc_tot.to_csv('mock_lc.csv', index=True)

make_full_sample(data, lc)
