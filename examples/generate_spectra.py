from mocksurvey.simulations import (
    get_luminosity_distance,
    get_effective_wavelength,
    get_snr,
    build_band_index,
    discard_small_snr,
    discard_skysurvey_flux
)
from skysurvey import DataSet
# from src import SNeIa_full_bgs, extract_ztf
from src import extract_snls
import pickle
import numpy as np
import logging
import sncosmo
import pandas
try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x:x



def eliminate_points_not_in_model(output, targets):
    logging.info('Eliminating points not in model')
    def checkwave(b, minwave, maxwave):
        band = sncosmo.get_bandpass(b)
        return band.minwave() >= minwave and band.maxwave() <= maxwave
    sncosmo_model = targets.template.sncosmo_model
    passcheck = np.zeros(len(output))
    unique_idx = np.unique(output['snid'].values)
    for index in unique_idx:
        mask = output['snid'] == index
        suboutput = output.loc[mask]
        bands_unique, band_index = build_band_index(suboutput.band.values)
        sncosmo_model.set(z = suboutput.z.values[0],
                          t0 = suboutput.t0.values[0],
                          x0 = suboutput.x0.values[0],
                          x1 = suboutput.x1.values[0],
                          c = suboutput.c.values[0],
                          mwebv = suboutput.mwebv.values[0])
        minwave = sncosmo_model.minwave()
        maxwave = sncosmo_model.maxwave()
        passcheck_unique = np.array([checkwave(band, minwave, maxwave) for band in bands_unique])
        passcheck[mask] = passcheck_unique[band_index]
    output['passcheck'] = pandas.Series(passcheck, index=output.index)
    return output.loc[output.passcheck.values == 1]

def get_trueflux_and_amplitude(output, targets):
    logging.info('Getting true fluxes and amplitudes')
    sncosmo_model = targets.template.sncosmo_model
    fluxtrue = np.zeros(len(output))
    amplitudes = np.zeros(len(output))
    unique_idx = np.unique(output['snid'].values)
    for index in unique_idx:
        mask = output['snid'] == index
        output_to_fit = output.loc[mask]
        band_to_fit = output_to_fit['band'].to_list()
        time_to_fit = output_to_fit['time'].to_list()
        zp_to_fit = output_to_fit['zp'].to_list()
        zpsys_to_fit = output_to_fit['zpsys'].to_list()
        time_at_max = output_to_fit['t0'].to_list()
        sncosmo_model.set(z = output_to_fit.z.values[0],
                          t0 = output_to_fit.t0.values[0],
                          x0 = output_to_fit.x0.values[0],
                          x1 = output_to_fit.x1.values[0],
                          c = output_to_fit.c.values[0],
                          mwebv = output_to_fit.mwebv.values[0])
        fluxtrue[mask] = sncosmo_model.bandflux(band_to_fit,
                                                time_to_fit,
                                                zp_to_fit,
                                                zpsys_to_fit)
        amplitudes[mask] = sncosmo_model.bandflux(band_to_fit,
                                                  time_at_max,
                                                  [30 for i in range(len(output_to_fit))],
                                                  zpsys_to_fit)
    output['fluxtrue'] = pandas.Series(fluxtrue, index=output.index)
    output['trueA'] = pandas.Series(amplitudes, index=output.index)
    return output

def f(x, a, b, c):
    """ """
    return a * x**2 + b * x + c


def generate_mock_sample(mock_survey):
    """
    Generates the mock sample based on the generated mock survey.

    !! Need to generate the mock sample first

    Parameters:
    ------------
    mock_ztf_path : str
        Path to the generated ztf mock sample

    Returns:
    ------------
    : pandas.DataFrame
        Mock spectra

    """
    data_sim = pandas.DataFrame(columns=["spec","time","wavelength","flux","fluxerr","valid","exptime","snid","flux_true","x0","x1","c","t0","mwebv","z"],
                               dtype=('int','float'))

    idx = mock_survey["fluxtrue"] != 0
    mock_survey = mock_survey[idx]

    mock_snid = np.unique(mock_survey["snid"])

    length_wave = np.array([218, 437, 3258, 656, 868])
    popt = np.array([-7.87649259e-07, 1.08382223e-02, -2.44390307e01])
    a, b, c = popt
    for i in tqdm(range(len(mock_snid))):
        mock_sn = mock_survey[mock_survey["snid"] == mock_snid[i]]
        
        x0 = mock_sn["x0"].mean()
        x1 = mock_sn["x1"].mean()
        c = mock_sn["c"].mean()
        t0 = mock_sn["t0"].mean()
        z = mock_sn["z"].mean()
        mwebv = mock_sn["mwebv"].mean()

        wave_size = np.random.choice(length_wave)
        min_wave = np.random.uniform(2900.0, 3400.0)
        max_wave = np.random.uniform(9000.0, 11000.0)
        wavelength = np.linspace(min_wave, max_wave, wave_size)
        
        mjd = np.random.choice(mock_sn["time"][mock_sn["time"].between(t0 - 10, t0 + 40)])

        source = source=sncosmo.SALT2Source(modeldir='./data/SALT_snf', m0file='nacl_m0_test.dat', m1file='nacl_m1_test.dat', clfile='nacl_color_law_test.dat')
        model= sncosmo.Model(source=source,effects=[sncosmo.CCM89Dust()],effect_names=['mw'],effect_frames=['obs'])
        p = {"z": z, "t0": t0, "x0": x0, "x1": x1, "c": c, "mwebv": mwebv, "mwr_v": 3.1}
        model.set(**p)

        min_w = model.minwave()
        max_w = model.maxwave()

        try:
            flux_true = model.flux(mjd, wavelength)
        except:
            try:
                wavelength = wavelength[wavelength < model.maxwave()]
                flux_true = model.flux(mjd, wavelength)
            except:
                wavelength = wavelength[wavelength > model.minwave()]
                flux_true = model.flux(mjd, wavelength)

        norm_values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1e12])
        norm = np.random.choice(norm_values)
        flux_true *= norm
        flux_err = (flux_true / f(wavelength, a, b, c)) * np.random.randn(
            len(flux_true)
        )
        flux = flux_true + 0.5 * flux_err

        n = len(flux)
        data_sim = pandas.concat(
            [data_sim,
             pandas.DataFrame(
                    {
                        "spec": np.repeat(i, n),
                        "time": np.repeat(mjd, n),
                        "wavelength": wavelength,
                        "flux": flux,
                        "fluxerr": np.abs(flux_err),
                        "valid": np.ones(n),
                        "exptime": np.repeat(np.nan, n),
                        "snid": np.repeat(mock_snid[i], n),
                        "flux_true": flux_true,
                        "x0": np.repeat(x0, n),
                        "x1": np.repeat(x1, n),
                        "c": np.repeat(c, n),
                        "t0": np.repeat(t0, n),
                        "mwebv": np.repeat(mwebv, n),
                        "z": np.repeat(z, n),
                    }
                )
            ]
        )
    
    return data_sim


if __name__=='__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", dest="filename", default="outdir/SN_Uchuu_modelcov.pkl", type=str)
    args = vars(parser.parse_args())

    with open('outdir/dataset_snls.pkl', "rb") as file:
        data = pickle.load(file)
        lc = pickle.load(file)

    survey=extract_snls()
    dset = DataSet(lc)
    dset.set_survey(survey)
    
    import skysurvey
    snia = skysurvey.SNeIa.from_data(data)
        
    snia.update_model(t0={"func":np.random.uniform, 'kwargs':{'low':survey.date_range[0], 'high':survey.date_range[1]}},
                      redshift={"kwargs": {'zmax':1.6,}, 'as':'z'},
                      mwebv={"func": skysurvey.effects.milkyway.get_mwebv, "kwargs": {"ra": "@ra", "dec": "@dec"}},)

    source = sncosmo.SALT2Source(modeldir='./data/SALT_snf', m0file='nacl_m0_test.dat', m1file='nacl_m1_test.dat', clfile='nacl_color_law_test.dat')
    model= sncosmo.Model(source=source,effects=[sncosmo.CCM89Dust()],effect_names=['mw'],effect_frames=['obs'])

    snia.set_template(model)
    dset.set_targets(snia)
    dset.data.index = dset.data.index.get_level_values(0)
    dset.data.index = dset.data.index.set_names('index')
    
    detected = dset.targets.data[dset.targets.data['valid']==1].index

    output = dset.data.loc[detected].join(dset.targets.data.loc[detected].drop(columns='template'))
    output['snid'] = output.index


    output = get_luminosity_distance(output, snia)
    output = get_effective_wavelength(output)
    output = eliminate_points_not_in_model(output, snia)
    output = get_trueflux_and_amplitude(output, snia)
    output = get_snr(output)
    output = discard_small_snr(output, threshold=0)
    output = discard_skysurvey_flux(output)

    output['snid'] = output.index

    mock_spectra = generate_mock_sample(output)
    mock_spectra.rename(columns={"snid":"sn", "flux_true":"fluxtrue"}, inplace=True)
    mock_spectra.to_csv("data/Mock_sp.csv")