from skysurvey.target.snia import SNeIaMagnitude, SNeIaColor, SNeIaStretch
from skysurvey.target import Transient
from skysurvey.effects import dust
import numpy as np
import healpy
from scipy.stats import gaussian_kde

from .load import load_bgs


def rand_ztf_positions(maps, size=1, zcut=0.1):
    ztf_map, bgs_redshifts = maps
    ztf_nside = healpy.npix2nside(len(ztf_map))
    bgs_nside = healpy.npix2nside(len(bgs_nside))

    sampled_pix = np.random.choice(
        np.arange(healpy.nside2npix(nside)), size=size, p=ztf_map
    )

    bgs_sampled_pix = healpy.ang2pix(bgs_nside, *healpy.pix2ang(ztf_nside, spl))

    for angpix, zpix in zip(sampled_pix, bgs_sampled_pix):
        new_ra, new_dec = draw_from_pixel(angpix, ztf_nside)
        ra.append(new_ra)
        dec.append(new_dec)

        kde = gaussian_kde(bgs_redshifts[zpix])
        z.append(kde.resample(1)[0][0])

    return (ra, dec, z)


def draw_from_pixel(pix, nside):
    boundaries_theta, boundaries_ra = healpy.vec2ang(healpy.boundaries(nside, pix).T)
    boundaries_dec = np.pi / 2 - boundaries_theta
    boundaries_ra = boundaries_ra[
        (boundaries_dec != -np.pi / 2) & (boundaries_dec != np.pi / 2)
    ]

    min_, max_ = np.min(boundaries_ra), np.max(boundaries_ra)
    if max_ - min_ >= 2 * np.pi - np.pi / nside:
        boundaries_ra[boundaries_ra < np.pi] += 2 * np.pi
    min_, max_ = np.min(boundaries_ra), np.max(boundaries_ra)

    in_pix = False
    while not in_pix:
        draw_ra = np.random.uniform(min_, max_)
        draw_ra -= 2 * np.pi * (draw_ra > 2 * np.pi)
        draw_dec = np.arcsin(
            np.random.uniform(
                np.min(np.sin(boundaries_dec)), np.max(np.sin(boundaries_dec))
            )
        )
        in_pix = pix == healpy.ang2pix(nside, 0.5 * np.pi - draw_dec, draw_ra)

    return np.array([draw_ra, draw_dec])


class SNeIa_ZTF_like(Transient):

    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _RATE = 2.35 * 10**4  # Perley 2020

    # {'model': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict(
        redshift={"kwargs": {"zmax": 0.2}},
        x1={"func": SNeIaStretch.nicolas2021},
        c={"func": SNeIaColor.intrinsic_and_dust},
        magabs={
            "func": SNeIaMagnitude.tripp1998,
            "kwargs": {"x1": "@x1", "c": "@c", "mabs": -19.3, "sigmaint": 0.10},
        },
        magobs={
            "func": "magabs_to_magobs",  # defined in Target (mother of Transients)
            "kwargs": {"z": "@z", "magabs": "@magabs"},
        },
        x0={
            "func": "magobs_to_amplitude",  # defined in Transients
            "kwargs": {"magobs": "@magobs", "param_name": "x0"},
        },  # because it needs to call sncosmo_model.get(param_name)
        radecz={
            "func": rand_ztf_positions,
            "kwargs": {"maps": load_maps(), "zcut": 0.06},
            "as": ["ra", "dec", "z", "t0"],
        },
        mwebv={"func": dust.get_mwebv, "kwargs": {"ra": "@ra", "dec": "@dec"}},
    )
