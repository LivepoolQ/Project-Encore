from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class RangerArgs(EmptyArgs):

    @property
    def Kc(self) -> int:
        """

        """
        return self._arg('Kc', 20, argtype=STATIC)
