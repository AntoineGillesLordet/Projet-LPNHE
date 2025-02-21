import numpy as np
import re,os
import sncosmo
from astropy.table import Table,vstack
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import argrelextrema
from scipy.stats import chi2
from scipy.interpolate import interp1d

from lemaitre import bandpasses
filterlib = bandpasses.get_filterlib()

import warnings
from iminuit.warnings import IMinuitWarning

warnings.filterwarnings("ignore", category=IMinuitWarning)
from tqdm.auto import tqdm
import logging
import os

#read SN
data_ztf=pd.read_csv('../data/pets_uchuu/z0.1/mock_sne.csv')
lc_ztf=pd.read_csv('../data/pets_uchuu/z0.1/mock_lc.csv')
data_ztf=data_ztf.sort_values(by=['name'])

Tgrid_folder = '/pscratch/sd/a/agillesl/Documents/Projet_LPNHE/Tgrids/pets_uchuu_0.1/'

modelpath='../data/SALT_snf/'
m0file='nacl_m0_test.dat'
m1file='nacl_m1_test.dat'
clfile='nacl_color_law_test.dat'

def _one_fit(t0, model):
    import warnings
    from iminuit.warnings import IMinuitWarning

    warnings.filterwarnings("ignore", category=IMinuitWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from lemaitre import bandpasses
    from logging import getLogger, WARNING
    getLogger("lemaitre.bandpasses.flibs").setLevel(WARNING)

    filterlib = bandpasses.get_filterlib()
    
    model.set(t0=t0)
    try:
        result, fitted_model = sncosmo.fit_lc(lc_sncosmo, model,['x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)
        return np.array([t0,
            result.parameters[2],
            result.errors['x0'],
            result.parameters[3],
            result.errors['x1'],
            result.parameters[4],
            result.errors['c'], 
            result.chisq,
            result.ndof])
    except:
        logging.info('No result for T0=%s'%t0)
        return None

names=np.unique(lc_ztf["name"])

with Parallel(n_jobs=256) as parallel:
    for name in tqdm(names, desc='Treating SN'):
        if os.path.exists(Tgrid_folder+'%s.dat'%name):
            continue
        #Select data for specific SN
        lc_sn=lc_ztf[lc_ztf.name==name]
        #Redshift and MW extinction
        mwebv=data_ztf[data_ztf.name==name].mwebv.values[0]
        zsn=data_ztf[data_ztf.name==name].zhel.values[0]
        #Transform to Table for sncosmo
        lc_sncosmo=Table.from_pandas(lc_sn[['mjd','band','flux','fluxerr','zp','magsys','valid']])
        #
        lc_sncosmo=lc_sncosmo[lc_sncosmo['valid']==1]
        # create a model
        source = sncosmo.SALT2Source(modeldir=modelpath,m0file=m0file, m1file=m1file, clfile=clfile)
        dust = sncosmo.CCM89Dust()
        model= sncosmo.Model(source=source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])

        model.set(z=zsn,mwebv=mwebv,mwr_v=3.1)  # set the model's redshift and MW
        try:
            # First fit
            res, mod = sncosmo.fit_lc(lc_sncosmo, model,['t0', 'x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)
            # If first fit succeeded, creates the Tmax grid and parallelize the sncosmo fitting
            T0_mcmc=np.concatenate([np.arange(res.parameters[1]-20,res.parameters[1]-5,0.1),
                                    np.arange(res.parameters[1]-5,res.parameters[1]+5,0.01),
                                    np.arange(res.parameters[1]+5,res.parameters[1]+20,0.1)])

            results = list(parallel(delayed(_one_fit)(t0, model) for t0 in tqdm(T0_mcmc, leave=False, desc=f'SN {name}')))

            # Save to data file as usual
            np.savetxt(Tgrid_folder + '%s.dat' %(name),
                       np.array([x for x in results if x is not None]),
                       fmt='%s',
                       header=' SN: %s, z:%0.3f, mwebv:%0.3f\n t0 x0 ex0 x1 ex1 c ec chisq dof'%(name,zsn,mwebv))


        except:
            # If it fails, save that there is no data and go to the nex SN
            np.savetxt(Tgrid_folder + '%s.dat' %(name),[0] ,fmt='%s', header='No data for this SN')
            logging.info('no data for %s'%name)
            continue
        
