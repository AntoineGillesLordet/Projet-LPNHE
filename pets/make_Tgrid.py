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

#read SN
data_ztf=pd.read_csv('mock_sne.csv')
lc_ztf=pd.read_csv('mock_lc.csv')
data_ztf=data_ztf.sort_values(by=['name'])

modelpath='../data/SALT_snf/'
m0file='nacl_m0_test.dat'
m1file='nacl_m1_test.dat'
clfile='nacl_color_law_test.dat'

def Tmax_grid(name):

    from lemaitre import bandpasses
    filterlib = bandpasses.get_filterlib()

    import warnings
    from iminuit.warnings import IMinuitWarning

    warnings.filterwarnings("ignore", category=IMinuitWarning)

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
    #First fit to get a starting position for MCMC
    try:
        res, mod = sncosmo.fit_lc(lc_sncosmo, model,['t0', 'x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)

        header=' SN: %s, z:%0.3f, mwebv:%0.3f\n t0 x0 ex0 x1 ex1 c ec chisq dof'%(name,zsn,mwebv)
        #Create array of T0
        T0_mcmc=np.concatenate([np.arange(res.parameters[1]-20,res.parameters[1]-5,0.1),np.arange(res.parameters[1]-5,res.parameters[1]+5,0.01),np.arange(res.parameters[1]+5,res.parameters[1]+20,0.1)])
        t0_chisq=[]
        x0=[]
        x1=[]
        c=[]
        ex0=[]
        ex1=[]
        ec=[]
        chisq=[]
        ndof=[]
        for i in range(np.size(T0_mcmc)):
            model= sncosmo.Model(source=source,effects=[dust],effect_names=['mw'],effect_frames=['obs'])
            model.set(z=zsn,t0=T0_mcmc[i],mwebv=mwebv,mwr_v=3.1)  # set the model's redshift and MW
            try:
                result, fitted_model = sncosmo.fit_lc(lc_sncosmo, model,['x0', 'x1', 'c'],bounds={'x0':(-0.1,10),'x1':(-5, 5),'c':(-3, 3)},phase_range=None,modelcov=False)
                t0_chisq.append(T0_mcmc[i])
                x0.append(result.parameters[2])
                x1.append(result.parameters[3])
                c.append(result.parameters[4])
                ex0.append(result.errors['x0'])
                ex1.append(result.errors['x1'])
                ec.append(result.errors['c'])
                chisq.append(result.chisq)
                ndof.append(result.ndof)



            except:
                print('No result for T0=%s'%T0_mcmc[i])


        grid_result=np.zeros((np.size(chisq),9))
        grid_result[:,0]=t0_chisq
        grid_result[:,1]=x0
        grid_result[:,2]=ex0
        grid_result[:,3]=x1
        grid_result[:,4]=ex1
        grid_result[:,5]=c
        grid_result[:,6]=ec
        grid_result[:,7]=chisq
        grid_result[:,8]=ndof


        np.savetxt('Tgrid/%s.dat' %(name),grid_result,fmt='%s', header=header)
    except:

        np.savetxt('Tgrid/%s.dat' %(name),[0] ,fmt='%s', header='No data for this SN')
        print('no data for %s'%name)


try:
    from tqdm.auto import tqdm
except:
    def tqdm(*args,**kwargs):
        pass

name=data_ztf.name.values
Parallel(n_jobs=256)(delayed(Tmax_grid)(nn) for nn in tqdm(name))

