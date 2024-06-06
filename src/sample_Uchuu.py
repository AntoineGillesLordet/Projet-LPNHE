from skysurvey.target.snia import SNeIaMagnitude, SNeIaColor, SNeIaStretch
from skysurvey.target import Transient
from skysurvey.effects import dust
import sncosmo
import numpy as np
from .load import load_bgs


def rand_positions(positions, size=1, weights=None, zcut=0.1):
    positions_ = positions[positions["z"] < zcut]
    index = np.random.choice(positions_.index, size=size, p=weights)

    return (
        positions_.loc[index].reset_index()["ra"],
        positions_.loc[index].reset_index()["dec"],
        positions_.loc[index].reset_index()["z"],
        np.array(index, dtype=int),
        positions_.loc[index].reset_index()["z_cosmo"],
    )


class SNeIa_full_bgs(Transient):

    _KIND = "SNIa"
    _RATE = 2.35 * 10**4  # /yr/Gpc^3 Perley 2020

    def __init__(self, path=None, filename="Uchuu.csv"):
        super().__init__()
        # {'model': func, 'prop': dict, 'input':, 'as':}
        self.set_model(dict(
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
                "kwargs": {"z": "@z_cosmo", "magabs": "@magabs"},
            },
            x0={
                "func": "magobs_to_amplitude",  # defined in Transients
                "kwargs": {"magobs": "@magobs", "param_name": "x0"},
            },  # because it needs to call sncosmo_model.get(param_name)
            radecz={
                "func": rand_positions,
                "kwargs": {"positions": load_bgs(path=path, filename=filename), "zcut": 0.06},
                "as": ["ra", "dec", "z", "bgs_id", "z_cosmo"],
            },
            mwebv={"func": dust.get_mwebv, "kwargs": {"ra": "@ra", "dec": "@dec"}},
        ))
        
        source = sncosmo.SALT2Source(modeldir='./data/SALT_snf', m0file='nacl_m0_test.dat', m1file='nacl_m1_test.dat', clfile='nacl_color_law_test.dat')
        model= sncosmo.Model(source=source,effects=[sncosmo.CCM89Dust()],effect_names=['mw'],effect_frames=['obs'])
        self.set_template(model)