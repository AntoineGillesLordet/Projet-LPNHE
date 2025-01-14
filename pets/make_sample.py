import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob,os,re
import sncosmo
from astropy.table import Table,vstack
from scipy.optimize import leastsq,minimize
import math
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
import saltworks
import pickle
from astropy.coordinates import SkyCoord, SphericalRepresentation, EarthLocation
from dustmaps.config import config
from astropy import units as u

from lemaitre import bandpasses
filterlib = bandpasses.get_filterlib()

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
config.reset()
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



with open("../outdir/dataset_snls.pkl", 'rb') as f:
    data = pickle.load(f)
    lc = pickle.load(f)

def create_ztf_lc_tds(data,lc):
    lc_tot=pd.DataFrame(columns=['sn','name', 'mjd', 'flux', 'fluxerr', 'band', 'zp', 'zpsys'])

    mjd,fl,efl,band,name,zp,lcs=[],[],[],[],[],[],[]


    non_det=0
    for sn in tqdm(data.index.values, desc='Creating lc'):
        tmax_phot=data.loc[sn].t0
        #DL photometry and apply quality cut and mjd
        try:
            lc_data=lc.loc[sn]

            #Select only the epochs around max
            lc_data=lc_data[abs(lc_data.time-tmax_phot)<50]

            lc_detection=lc_data[lc_data.flux/lc_data.fluxerr>5]

            if (np.size(np.unique(lc_detection['band']))>=2) and (np.size(lc_detection['flux'])>=5):

                band.append(lc_data['band'].values)
                fl.append(lc_data['flux'].values)
                efl.append(lc_data['fluxerr'].values)
                mjd.append(lc_data['time'].values)
                zp.append(lc_data['zp'].values)

                name.append(np.zeros(np.shape(lc_data)[0]).astype(int)+sn)
                lcs.append(np.zeros(np.shape(lc_data)[0]).astype(int)+sn)
            else:
                non_det=non_det+1
        except:
            pass
    lc_tot['mjd']=np.concatenate(mjd)
    lc_tot['flux']=np.concatenate(fl)
    lc_tot['fluxerr']=np.concatenate(efl)
    lc_tot['zp']=np.concatenate(zp)
    lc_tot['band']=np.concatenate(band)



    lc_tot['name']=np.concatenate(name)
    lc_tot['sn']=np.concatenate(lcs)

    lc_tot.insert (5, 'magsys', 'AB')
    lc_tot.insert (6, 'exptime', 'NaN')
    lc_tot.insert (7, 'valid', 1)
    lc_tot.insert (8, 'lc', 0)
    lc_tot.insert (10, 'mag_sky', 'NaN')
    lc_tot.insert (11, 'seeing', 'NaN')
    lc_tot=lc_tot.reset_index()
    lc_tot=lc_tot.drop(columns=['index'])

    # light curve indexation
    c = 0
    id_lc = np.ones(len(lc_tot['flux']))  # .astype(int)

    for i in tqdm(range(lc_tot['sn'].iloc[-1]+1), desc="Indexing LC"):
        idx_sn = lc_tot['sn'] == i
        lcs = lc_tot[idx_sn]
        _, idx = np.unique(lcs["band"], return_index=True)
        for bd_sn in lcs['band'].iloc[np.sort(idx)]:
            id_lc[(lc_tot['sn'] == i) & (lc_tot['band'] == bd_sn)] = c  # [c]*len(lc[lc]))
            c += 1

    id_lc = np.hstack(np.array(id_lc))
    lc_tot['lc']=id_lc.astype(int)
    
    float32_cols = list(lc_tot.select_dtypes(include='float32'))
    lc_tot[float32_cols] = lc_tot[float32_cols].astype('float64')

    lc_tot.to_csv('mock_lcs.csv', index=False)
    return lc_tot




### DATA file


def get_SN_data(data):	
    data=data[['z','ra','dec','t0','x0','x1','c']]

    data['name']=data.index.values
    data['sn']=data.index.values
    data['classification']='SNIa'
    data['comments']=''

    data=data.rename(columns={'z': 'zhel', 't0': 'tmax'})              

    data['mwebv']=get_mwebv(data['ra'].values, data['dec'].values, dustmap='sf11')

    data['zcmb']=zcmb_tot(data['zhel'],data['ra'], data['dec'])

    data.to_csv('mock_sne.csv', index=False)
    return data

def remove_outlier_fit(name, lc, sne, sigma=3):

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
    model= sncosmo.Model(source=source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])

    model.set(z=zsn,mwebv=mwebv,mwr_v=3.1)  # set the model's redshift and MW
    #First fit to get a starting position for MCMC
    res, mod = sncosmo.fit_lc(lc_sncosmo, model, ['t0', 'x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)

    residu=0
    counter=0
    progress = tqdm(desc="Recursive execution", leave=False)
    while np.size(residu)>0:

        counter += 1
        if counter == 5:
            break
        residu=[]
        for i,j in enumerate(np.unique(lc_sncosmo[lc_sncosmo['valid']==1]['band'])):

            lc_band=lc_sncosmo[(lc_sncosmo['band']==j) &(lc_sncosmo['valid']==1)]


            residual=lc_band['flux']-mod.bandflux(j, lc_band['mjd'], zp=lc_band['zp'], zpsys=lc_band['magsys'])
            #3sigma clipping
            idx=np.where(np.abs(residual-np.mean(residual))/np.std(residual)>sigma)[0]


            pos_band=min(np.where(lc_sncosmo[(lc_sncosmo['valid']==1)]['band']==j)[0])				
            idx=idx+pos_band


            residu.append(idx)
        residu=np.concatenate(residu)
        lc_sncosmo['valid'][residu]=0

        res, mod = sncosmo.fit_lc(lc_sncosmo[lc_sncosmo['valid']==1], model,['t0', 'x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)
        progress.update()
    return lc_sncosmo



def make_full_sample(data,lcdata):
    #Create ZTf sne training dataset  
    sne = get_SN_data(data)

    #Create ZTf spectra training dataset  
    lc = create_ztf_lc_tds(data,lcdata)

    lc_tot = Table()
    #Remove photometric outliers:
    for i,sn in tqdm(enumerate(np.unique(lc.name)), desc="Removing outlier"):
        lc_out=remove_outlier_fit(sn, lc, sne, sigma=3)
        lc_tot=vstack([lc_tot, lc_out])
    lc_tot=pd.DataFrame(np.array(lc_tot),columns=['sn', 'name', 'mjd', 'flux', 'fluxerr', 'magsys', 'exptime', 'valid','lc', 'band', 'mag_sky', 'seeing', 'zp'])
    lc_tot.to_csv('mock_lc.csv', index=True)

make_full_sample(data, lc)


