import jax

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

from astropy.cosmology import Planck15, FlatLambdaCDM
Planck15 = FlatLambdaCDM(name='Planck15', H0=Planck15.H0, Om0=0.3089, Tcmb0=Planck15.Tcmb0, Neff=Planck15.Neff, m_nu=Planck15.m_nu, Ob0=Planck15.Ob0)

from .load import *
from .plot import *
from .fit import *
from .sample_Uchuu import *
from .tools import *
from .logging import *

