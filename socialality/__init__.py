import qpid

from .__args import SocialalityArgs
from .model import Socialality, SocialalityModel

qpid.register(sa=[Socialality, SocialalityModel])
qpid.register_args(SocialalityArgs, 'sa Args')
