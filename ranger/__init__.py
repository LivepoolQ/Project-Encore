import qpid

from .__args import RangerArgs
from .model import Ranger, RangerModel

qpid.register(enc=[Ranger, RangerModel])
qpid.register_args(RangerArgs, 'Ranger Args')
