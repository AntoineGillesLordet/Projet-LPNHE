from skysurvey import DataSet
# from src import SNeIa_full_bgs, extract_ztf
# from src import extract_snls
import pickle
import numpy as np
import logging
import sncosmo
import pandas
try:
    from tqdm.auto import tqdm
except:
    tqdm = lambda x:x

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
    data_sim = []
    
    idx = mock_survey["fluxtrue"] != 0
    output = mock_survey[idx]
    
    mock_snid = np.unique(output["snid"])
    
    length_wave = np.array([218, 437, 3258, 656, 868])
    popt = np.array([-7.87649259e-07, 1.08382223e-02, -2.44390307e01])
    a, b, c = popt
    for i,snid in tqdm(enumerate(mock_snid), total=len(mock_snid)):
        try:
            mock_sn = output[output["snid"] == snid]
            x0 = mock_sn.iloc[0]["x0"]
            x1 = mock_sn.iloc[0]["x1"]
            c = mock_sn.iloc[0]["c"]
            t0 = mock_sn.iloc[0]["t0"]
            z = mock_sn.iloc[0]["z"]
            mwebv = mock_sn.iloc[0]["mwebv"]
            
            wave_size = np.random.choice(length_wave)
            min_wave = np.random.uniform(2900.0, 3400.0)
            max_wave = np.random.uniform(9000.0, 11000.0)
            wavelength = np.linspace(min_wave, max_wave, wave_size)
            
            
            mjd = np.random.choice(mock_sn["time"][mock_sn["time"].between(t0 - 10, t0 + 40)])
            
            source = sncosmo.get_source('salt2', version='2.4')
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
            
            spec_sim = pandas.DataFrame(columns=["spec","time","wavelength","flux","fluxerr","valid","exptime","snid","flux_true","x0","x1","c","t0","mwebv","z"],
                                       dtype=('int','float'))
            spec_sim['wavelength'] = wavelength
            spec_sim['flux'] = flux
            spec_sim['flux_true'] = flux_true
            spec_sim['fluxerr'] = np.abs(flux_err)
            spec_sim['time'] = mjd
            spec_sim['valid'] = 1
            spec_sim['exptime'] = np.nan
            spec_sim['snid'] = snid
            spec_sim['spec'] = i
            spec_sim["z"] = z
            spec_sim["x0"] = x0
            spec_sim["x1"] = x1
            spec_sim["c"] = c
            spec_sim["t0"] = t0
            spec_sim["mwebv"] = mwebv
            data_sim.append(spec_sim)    
        except:
            continue
    return pandas.concat(data_sim)

if __name__=='__main__':
    from src import load_from_skysurvey
    from mocksurvey.simulations import (
        get_luminosity_distance,
        get_effective_wavelength,
        get_snr,
        build_band_index,
        discard_small_snr,
        discard_skysurvey_flux,
        eliminate_points_not_in_model,
        get_trueflux_and_amplitude,
        get_sn_position)
    
    data, lc = load_from_skysurvey("/pscratch/sd/a/agillesl/Documents/Projet_LPNHE/SN_dataset/dataset_hsc.pkl", survey='HSC', return_lc=True)


    lc.reset_index(drop=True, inplace=True)
    data.set_index("sn", drop=False, inplace=True)
    # Select only points at 5 sigma and in [tmax-50, tmax+100]
    lc = lc[(lc.flux/lc.fluxerr>5) &
                    (lc.mjd.between(data.loc[lc.sn, 'tmax'].reset_index(drop=True) - 50,
                                         data.loc[lc.sn, 'tmax'].reset_index(drop=True) + 100))].copy()
    lc.reset_index(drop=True, inplace=True)
    
    # Select SN on : >=5 detections in >=2 bands
    goods_sn = (lc.groupby(["sn"]).band.nunique() >= 2) & (lc.groupby(["sn"])['flux'].count() >= 5)
    
    lc = lc[goods_sn.loc[lc.sn].reset_index(drop=True)]
    
    data = data[data.sn.isin(lc.sn.unique())]


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
    mock_spectra.to_csv("../data/Mock_sp_test.csv")