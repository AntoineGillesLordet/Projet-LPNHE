import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--survey-name",
                    dest="survey_name",
                    type=str,
                    help="Survey to generate mocks for")

parser.add_argument("-S", "--survey-logs",
                    dest="survey_path",
                    default="./data",
                    type=str,
                    help="General path for survey logs")

parser.add_argument("-m", "--model-dir",
                    dest="model_dir",
                    default="./data/SALT_snf",
                    type=str,
                    help="Path for salt2 data files")

parser.add_argument("-o", "--outdir",
                    dest="outdir",
                    default="./outdir",
                    type=str,
                    help="Output directory")


args = vars(parser.parse_args())

import os
if not os.path.exists(args["outdir"]):
    raise FileNotFoundError(f'No dir {args["outdir"]} to output too, consider creating it with mkdir or changing it with -o/--outdir')

import skysurvey
import sncosmo
import pandas
import pickle
import numpy as np
from lemaitre import bandpasses
from shapely import geometry

## SURVEY LOADING
logging.info('Loading Survey')
if args["survey_name"]=='ztf':
    # ZTF is already stored as a skysurvey.survey object 
    with open(args["survey_path"] + '/ztf_survey.pkl', "rb") as file:
        survey = pickle.load(file)
        
elif args["survey_name"]=='snls':
    # SNLS is defined by pointing its fooprint at its 4 fields
    survey = skysurvey.GridSurvey.from_pointings(data=pandas.read_csv(args["survey_path"] + '/snls_obslogs_cured.csv', encoding='utf-8'),
                                               footprint=geometry.box(-0.5, -0.5, 0.5, 0.5),
                                               fields_or_coords={'D1': {'ra': 36.450190, 'dec': -4.45065},
                                                                 'D2': {'ra': 150.11322, 'dec': +2.21571},
                                                                 'D3': {'ra': 214.90738, 'dec': +52.6660},
                                                                 'D4': {'ra': 333.89903, 'dec': -17.71961}}
                                              )
    # The filters name needed to be replaced to work properly with bbf/bandpasses, not sure if it's still needed
    survey.data.replace({'MEGACAMPSF::g':'megacam6::g',
                       'MEGACAMPSF::i':'megacam6::i2',
                       'MEGACAMPSF::r':'megacam6::r',
                       'MEGACAMPSF::y':'megacam6::i2', # Hotfix as the y band was not implemented
                       'MEGACAMPSF::z':'megacam6::z'}, inplace=True)
    
elif args["survey_name"]=='hsc':
    # Same as SNLS, except HSC has only one field
    survey = skysurvey.Survey.from_pointings(pandas.read_csv(args["survey_path"] + '/hsc_logs_realistic_skynoise.csv', index_col=0), 
                                             geometry.Point(0,0).buffer(0.7))
elif args["survey_name"]:
    raise ValueError(f'Mock generation not yet implemented for survey {survey_name}')
else:
    raise ValueError('Survey name not provided, use the -s or --survey-name option')


## GENERAL TARGETS GENERATION
logging.info('Drawing SNIa')
snia = skysurvey.SNeIa()

# This loads Mahmoud's extended SALT2 model
source = sncosmo.SALT2Source(modeldir=args["model_dir"],
                             m0file='nacl_m0_test.dat',
                             m1file='nacl_m1_test.dat',
                             clfile='nacl_color_law_test.dat')
# and adds dust extinction in observer frame (it will be the MW one)
model= sncosmo.Model(source=source,
                     effects=[sncosmo.CCM89Dust()],
                     effect_names=['mw'],
                     effect_frames=['obs'])
snia.set_template(model)

# Avoid drawing SN that are outside the survey time scale and add dust drawing

from skysurvey.effects.milkyway import mwebv_model
snia.update_model(t0={"func":np.random.uniform,
                      'kwargs':{'low':survey.date_range[0], 'high':survey.date_range[1]}}, 
                  redshift={"kwargs": {'zmax': 0.2 if args["survey_name"]=="ztf" else 1.6,}, 'as':'z'},
                  **mwebv_model)

if args["survey_name"] in ["hsc","snls"]:
    # For SNLS and HSC, we can avoid drawing SN outside their fields
    _ = snia.draw(tstart=survey.date_range[0], tstop=survey.date_range[1],
                  skyarea=survey.get_skyarea().buffer(0.01), zmax=1.6,
                  inplace=True)
else:
    # But not for ztf
    _ = snia.draw(tstart=survey.date_range[0], tstop=survey.date_range[1], zmax=0.2, inplace=True)
    
logging.info('Computing lightcurves')

dset = skysurvey.DataSet.from_targets_and_survey(snia, survey, incl_error=True) # incl_error=True adds gaussian noise to the LC using fluxerr

with open(args["outdir"] + f'/dataset_{args["survey_name"]}.pkl', 'wb') as file:
    pickle.dump(dset, file)

logging.info(f'Done, dataset has been saved to {args["outdir"]}/dataset_{args["survey_name"]}.pkl')