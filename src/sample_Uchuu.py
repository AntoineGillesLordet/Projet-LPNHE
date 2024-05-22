from skysurvey.target.snia import SNeIaMagnitude, SNeIaColor, SNeIaStretch
from skysurvey.target import Transient
from skysurvey.effects import dust
import numpy as np
from .load import load_bgs


def rand_positions(positions, size=1, weights=None, zcut=0.1):
    positions_ = positions[positions["z"] < zcut]
    index = np.random.choice(positions_.index, size=size, p=weights)

    return (
        positions_.loc[index].reset_index()["ra"],
        positions_.loc[index].reset_index()["dec"],
        positions_.loc[index].reset_index()["z"],
    )


class SNeIa_full_bgs(Transient):

    _KIND = "SNIa"
    _TEMPLATE = "salt2"
    _RATE = 2.35 * 10**4  # Perley 2020

    def __init__(self, path=None):
        # {'model': func, 'prop': dict, 'input':, 'as':}
        self.model = dict(
            redshift={"kwargs": {"zmax": 0.2}},
            x1={"func": SNeIaStretch.nicolas2021},
            c={"func": SNeIaColor.intrinsic_and_dust},
            t0={"func": np.random.uniform, "kwargs": {"low": 58179, "high": 59215}},
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
                "func": rand_positions,
                "kwargs": {"positions": load_bgs(path), "zcut": 0.06},
                "as": ["ra", "dec", "z"],
            },
            mwebv={"func": dust.get_mwebv, "kwargs": {"ra": "@ra", "dec": "@dec"}},
        )
