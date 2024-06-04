import argparse
import logging
import numpy as np
from src import (
    setup_logger,
    logger,
    extract_ztf,
    SNeIa_full_bgs,
    dset_sanitize_and_filter,
    fit_lc,
    sncosmo_to_edris,
    run_edris,
    get_cov_from_hess,
    fit_cosmo,
    corner_,
    mag_Planck18,
    )
from skysurvey import DataSet
import os
import pickle
import matplotlib.pyplot as plt
import jax.numpy as jnp

import warnings
from iminuit.warnings import IMinuitWarning
warnings.filterwarnings('ignore', category=IMinuitWarning)

# Parser to get the run number

parser = argparse.ArgumentParser(description="Take a SN sample from Uchuu, produce a survey with ZTF logs, fit the lightcurves and run edris. Also makes coffee")

parser.add_argument("-i", "--run-id", dest="run_nb", type=int)

args = vars(parser.parse_args())

dir_name = f"Run_{args['run_nb']:03d}"

# ouptut directory and logger setup

if not os.path.exists(f'outdir/{dir_name}'):
    os.mkdir(f'outdir/{dir_name}')
    
setup_logger(f'outdir/{dir_name}/{dir_name}.log')


# SN generation

if os.path.exists(f'outdir/{dir_name}/SN_data.pkl'):
    logger.log(logging.INFO, "Already generated SN sample, skipping step")
    with open(f'outdir/{dir_name}/SN_data.pkl', 'rb') as f:
        lc = pickle.load(f)
        data = pickle.load(f)
        try:
            results = pickle.load(f)
            meta = pickle.load(f)
        except:
            results, meta = None,None

        dset = DataSet(lc)
        logger.setLevel(logging.WARNING)
        dset.set_survey(extract_ztf())
        dset.set_targets(SNeIa_full_bgs.from_data(data))
        index = np.where(dset.targets.data['good'])[0]
        logger.setLevel(logging.DEBUG)
else:
    logger.log(logging.INFO, "Reading survey")
    survey = extract_ztf()

    logger.log(logging.INFO, "Generating SNe sample")
    snia = SNeIa_full_bgs()
    _ = snia.draw(tstart=survey.date_range[0], tstop=survey.date_range[1], inplace=True,  zmax=0.06)
    logger.log(logging.INFO, f"{len(snia.data)} SNe generated")

    logger.log(logging.INFO, "Generating lightcurves")
    dset = DataSet.from_targets_and_survey(snia, survey)

    logger.log(logging.INFO, "Correcting rate to observations")
    index = dset_sanitize_and_filter(dset)
    logger.log(logging.INFO, f"{len(index)} SNe left")

    logger.log(logging.INFO, "Saving dataset before fitting")
    with open(f'outdir/{dir_name}/SN_data.pkl', "wb") as f:
        pickle.dump(dset.data, f)
        pickle.dump(dset.targets.data, f)
        logger.log(logging.INFO, "Done")
    results, meta=None,None

n_bins=6


# Fitting lightcurves

if (results is not None and meta is not None):
    logger.log(logging.INFO, "Already generated lc fit, skipping step")
else:
    
    results, meta = fit_lc(dset, index, savefile=f'outdir/{dir_name}/SN_data.pkl')

    data = dset.targets.data
    index = data[data['converged']].index


# EDRIS
    
if os.path.exists(f'outdir/{dir_name}/edris_result.pkl'):
    logger.log(logging.INFO, "Already generated edris file, skipping step")
    with open(f"outdir/{dir_name}/edris_result.pkl", 'rb') as f:
        exp = pickle.load(f)
        obs = pickle.load(f)
        cov = pickle.load(f)
        res = pickle.load(f)
        cov_res = pickle.load(f)
        iter_params = pickle.load(f)
else:
    index=data[data['converged']].index
    exp, cov, obs = sncosmo_to_edris(results, data, index, n_bins=n_bins)

    res, hess, loss, iter_params = run_edris(obs, cov, exp, verbose=True)

    cov_res =  get_cov_from_hess(hess)

    logger.log(logging.INFO, f"Nombre de SN générées par skysurvey  :  {len(data)}")
    logger.log(logging.INFO, f"Nombre de SN générées (rate corrigé) :  {sum(data['keep'])}")
    logger.log(logging.INFO, f"Nombre de SN fittées avec sncosmo    :  {sum(data['good'])}")
    logger.log(logging.INFO, f"Nombre de SN dont le fit a convergé  :  {sum(data['converged'])}")
    logger.log(logging.INFO, f"Nombre de SN utilisées pour edris    :  {sum(data['used_edris'])}")


    logger.log(logging.INFO, "Saving edris result")

    with open(f"outdir/{dir_name}/edris_result.pkl", 'wb') as f:
        pickle.dump(exp, f)
        pickle.dump(obs, f)
        pickle.dump(cov, f)
        pickle.dump(res, f)
        pickle.dump(cov_res, f)
        pickle.dump(iter_params, f)


std_mag = obs.mag - jnp.matmul(res['coef'], res['variables'])


# Cosmo fit
    
n_var = 2

popt, pcov, mag_to_z_cosmo, z_to_mag = fit_cosmo(exp['z_bins'], res['mu_bins'], cov_res[n_var:n_bins+n_var,n_var:n_bins+n_var])

data['calc_z_cosmo'] = np.NaN
data.loc[data[data['used_edris']].index, 'calc_z_cosmo'] = mag_to_z_cosmo(std_mag)

logger.log(logging.INFO, "Last file checkpoint")

with open(f"outdir/{dir_name}/SN_data.pkl", 'wb') as f:
    pickle.dump(dset.data, f)
    pickle.dump(data, f)
    pickle.dump(results, f)
    pickle.dump(meta, f)

# PLOTS

### Event distribution
logger.log(logging.INFO, "Generating usual plots")
_=corner_(data[data['used_edris']], var_names=['x1','c','t0','ra','dec','z','magobs','magabs'])
plt.savefig(f"outdir/{dir_name}/SN_distribution.png")

### Hubble diagramm
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex='col', figsize=(7,6), gridspec_kw={'height_ratios': [3,1]})

ax1.errorbar(exp['z_bins'], res['mu_bins'], yerr=jnp.sqrt(jnp.diag(cov_res[2:2+n_bins,2:2+n_bins])), color='tab:blue', label='edris')
ax1.plot(np.linspace(5e-3, 0.06,1000), mag_Planck18(np.linspace(5e-3, 0.06,1000)), color='tab:green', linestyle=':', label='Planck18')
ax1.plot(np.linspace(5e-3, 0.06,1000), z_to_mag(np.linspace(5e-3, 0.06,1000), *popt), color='tab:red', label='fit')

ax1.scatter(exp["z"], std_mag, s=.5, alpha=.3, color='k', label='Standardised magnitudes')
ax1.legend()
ax1.set_ylabel(r'$\mu$')
fig.suptitle(r"Modèle $\Lambda CDM$ fitté sur diagramme de Hubble")

ax2.errorbar(exp['z_bins'], res['mu_bins']-mag_Planck18(exp["z_bins"]), yerr=jnp.sqrt(jnp.diag(cov_res[2:2+n_bins,2:2+n_bins])), color='tab:blue')
ax2.plot(np.linspace(5e-3, 0.06,1000), z_to_mag(np.linspace(5e-3, 0.06,1000), *popt) - mag_Planck18(np.linspace(5e-3, 0.06, 1000)), color='tab:red')
ax2.scatter(exp["z"], std_mag - mag_Planck18(exp["z"]), color='k', s=.5, alpha=.3)
lims = ax2.get_xlim()
ax2.hlines(0., xmin=lims[0], xmax=lims[1], color='tab:green', linestyle=':')
ax2.set_xlim(*lims)
ax2.set_ylabel(r'$\Delta\mu$')
ax2.set_xlabel(r'$z$')
plt.savefig(f'outdir/{dir_name}/Hubble.png')


### Peculiar velocities
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(15,10))
ax1.scatter(data[data["used_edris"]]["z_cosmo"],
            data[data["used_edris"]]['calc_z_cosmo'],
            s=1)
ax1.plot(np.linspace(1e-3, 0.07, 1000), np.linspace(1e-3, 0.07, 1000), ':k', label='baseline $z_{edris}=z_{uchuu}$')
ax1.set_ylabel(r'$z_{cosmo, edris}$')
fig.suptitle(r'Is the $z_{cosmo}$ reconstructed by edris the same as the uchuu one ?')

ax2.axis('off')

ax3.scatter(data[data["used_edris"]]["z_cosmo"],
            data[data["used_edris"]]['calc_z_cosmo'] - data[data["used_edris"]]["z_cosmo"],
            s=1)
ax3.plot(np.linspace(1e-3, 0.07, 1000), np.zeros(1000), ':k')
ax3.set_xlabel(r'$z_{cosmo, uchuu}$')
ax3.set_ylabel(r'$\Delta z_{cosmo}$')

delta_z_cosmo = data[data["used_edris"]]["z_cosmo"] - data[data["used_edris"]]['calc_z_cosmo']

ax4.hist(delta_z_cosmo, bins=60, orientation='horizontal')
m = ax4.get_xlim()[1]
ax4.hlines(delta_z_cosmo.mean(), xmin=0, xmax=m, color='r', label=fr'Mean; $\Delta z = {delta_z_cosmo.mean():.2e}$')
ax4.hlines([delta_z_cosmo.mean() - delta_z_cosmo.std(), delta_z_cosmo.mean() + delta_z_cosmo.std()], xmin=0, xmax=m, color='r', linestyle=':', label=fr'1 sigma; $\sigma = {delta_z_cosmo.std():.2e}$')
ax4.set_xlabel(r'Count')
fig.legend(loc=1, bbox_to_anchor=(0.4, 0.4, 0.5, 0.5))
plt.savefig(f'outdir/{dir_name}/z_cosmo.png')

