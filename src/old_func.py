"""
Only there "just in case", but I don't believe I will use it again.
"""

def dset_sanitize_and_filter(dset, return_index=True, all_pts=False):
    """
    Flags SN that are detected with the key ``keep`` and those that pass some cuts for lightcurve fitting with ``valid``.

    Parameters
    ----------
    dset : skysurvey.DataSet
        DataSet of the SN containing both the targets data and lightcurves
    return_index : bool, optional
        If True, returns the index of the SN that will be fitted as a list, otherwise only the dset.targets.data dataframe will be updated
    """
    logger.log(logging.INFO, "Correcting rate to observations and filtering based on observations")

    if all_pts:
        dset.data["detected"] = True
    else:
        dset.data["detected"] = (dset.data["flux"] / dset.data["fluxerr"]) > 5

    dset.targets.data["keep"] = False
    dset.targets.data["valid"] = False

    bands = np.unique(dset.data["band"])

    ids = np.unique(list(map(lambda x: x[0], dset.data.index)))
    for i in tqdm(ids):
        target = dset.targets.data.loc[i]
        obs_data = dset.data.loc[i]
        # Flags SN that were not observed to correct the rate
        dset.targets.data.loc[i, "keep"] = np.any(
            obs_data["time"].between(target["t0"] - 10, target["t0"] + 15)
        )

        dset.targets.data.loc[i, "valid"] = (
            dset.targets.data.loc[i, "keep"]  # SN should be observed
            and np.sum(
                [
                    np.sum(
                        obs_data[obs_data["detected"] & (obs_data["band"] == b)][
                            "time"
                        ].between(target["t0"] - 30, target["t0"] + 100)
                    )
                    >= 5
                    for b in bands
                ]
            ) >= 2 # Two bands should have 5 or more data points
            and (
                np.sum(
                    obs_data[obs_data["detected"]]["time"].between(
                        target["t0"] - 30, target["t0"]
                    )
                )
                > 1
            )  # At least one data point before t0
            and (
                np.sum(
                    obs_data[obs_data["detected"]]["time"].between(
                        target["t0"], target["t0"] + 100
                    )
                )
                > 1
            )  # At least one data point after t0
        )
    logger.log(logging.INFO, "Done")

    if return_index:
        return dset.targets.data[dset.targets.data["valid"]].index
    
def load_maps(ztf_path="data/ztf_distribution.fits",
            bgs_pix_path="data/bgs_pixels.pkl",
            ztf_nside=128,
            bgs_nside=128,
            F=0.1,
        ):

    logger.log(logging.INFO, "Loading ZTF skymap and BGS redshift distribution")
    try:
        map_ = healpy.read_map(ztf_path)

        with open(bgs_pix_path, "rb") as fp:
            bgs_redshifts = pickle.load(fp)

    except FileNotFoundError:
        logger.log(logging.INFO, "Maps file missing, computing them again and saving")

        # Initial SN density skymap
        ztf_sn = pandas.read_csv("data/data_ztf.csv", index_col=0)
        ids = healpy.ang2pix(
            theta=np.pi / 2 - ztf_sn["dec"] * np.pi / 180,
            phi=ztf_sn["ra"] * np.pi / 180,
            nside=ztf_nside,
        )
        map_ = np.zeros(healpy.nside2npix(ztf_nside))
        for i in tqdm(ids):
            map_[i] += 1
        map_ = healpy.smoothing(map_, fwhm=F)
        map_ -= map_.min()

        bgs_df = load_bgs(
            columns=[
                "RA",
                "DEC",
                "Z",
                "Z_COSMO",
                "STATUS",
                "V_PEAK",
                "V_RMS",
                "R_MAG_ABS",
                "R_MAG_APP",
            ]
        )

        # Mask of areas where there are no BGS galaxies
        id_bgs = healpy.ang2pix(
            theta=np.pi / 2 - bgs_df["dec"] * np.pi / 180,
            phi=bgs_df["ra"] * np.pi / 180,
            nside=nside,
        )
        mask = np.zeros(healpy.nside2npix(nside), dtype=bool)
        for i in tqdm(id_bgs):
            mask[i] = True
        map_[~mask] = 0

        # Normalization and saving
        map_ /= np.sum(map_)
        healpy.write_map("data/ztf_distribution.fits", map_)

        # list of BGS redshifts along lines of sight
        z_nside = 64
        id_bgs = healpy.ang2pix(
            theta=np.pi / 2 - bgs_df["dec"] * np.pi / 180,
            phi=bgs_df["ra"] * np.pi / 180,
            nside=bgs_nside,
        )
        bgs_pix = [[]] * healpy.nside2npix(bgs_nside)
        for i, nb_pix in tqdm(enumerate(id_bgs), total=len(id_bgs)):
            bgs_pix[nb_pix] = bgs_pix[nb_pix] + [id_bgs.index[i]]

        bgs_redshifts = [np.array(bgs_df.loc[ids]["z"]) for ids in bgs_pix]

        with open("data/bgs_redshifts_map.pkl", "wb") as fp:
            pickle.dump(bgs_redshifts, fp)
        logger.log(logging.INFO, "Done")

    finally:
        return map_, bgs_redshifts


def _2D_cmap(x,y, renorm=True):
    xx, yy = x.copy(), y.copy()
    if renorm:
        xx -= xx.min()
        yy -= yy.min()
        xx /= xx.max()
        yy /= yy.max()
    return np.array(list(zip(0.7*((xx.ravel()-1)**2 + (yy.ravel()-(1-0.5/np.sqrt(2)))**2),
                               0.7*((xx.ravel())**2 + (yy.ravel()-(1-0.5/np.sqrt(2)))**2),
                               ((xx.ravel()-0.5)**2 + (yy.ravel()+0.5)**2)/2.5))
                   )

def plot_flux_phwl(input_flux, model_flux, wl_rf, ph_rf, linthresh=1e-3):
    fig, axs = plt.subplots(ncols=2, figsize=(20,10))
    axs[0].scatter(input_flux,
                   model_flux,
                   marker='.',
                   s=5,
                   alpha=0.5,
                   c=_2D_cmap(wl_rf, ph_rf, renorm=False)
                  )
    min_ = min(input_flux.min(), model_flux.min())
    max_ = max(input_flux.max(), model_flux.max())
    ext = max(np.abs(min_), np.abs(max_))
    axs[0].plot([-ext, ext], [-ext, ext], 'k:')
    axs[0].plot([-ext, ext], [ext, -ext], 'k:')
    
    if linthresh:
        axs[0].set_yscale('symlog', linthresh=linthresh)
        axs[0].set_xscale('symlog', linthresh=linthresh)
    
    axs[0].set_xlabel("Input Flux")
    axs[0].set_ylabel("Model Flux")

    wl_lin, ph_lin = np.meshgrid(np.linspace(2000, 9000, 1000), np.linspace(-20, 50, 1000))

    axs[1].pcolormesh(wl_lin, ph_lin, _2D_cmap(wl_lin, ph_lin).reshape((1000,1000,3)))
    axs[1].set_ylabel("Phase")
    axs[1].set_xlabel("Wavelength")
    axs[1].set_ylim((50, -20))
