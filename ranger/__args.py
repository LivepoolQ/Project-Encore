import numpy as np

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class RangerArgs(EmptyArgs):

    @property
    def Kc(self) -> int:
        """

        """
        return self._arg('Kc', 20, argtype=STATIC)
    
    @property
    def mem_step(self) -> int:
        """
        
        """
        return self._arg('mem_step', 8, argtype=STATIC)
    
    @property
    def use_group(self) -> int:
        """
        Choose whether to use pedestrian groups when calculating SocialCircle.
        """
        return self._arg('use_group', 1, argtype=STATIC, desc_in_model_summary='use_group_model')
    
    @property
    def group_distance(self) -> int:
        return self._arg('group_distance', 6, argtype=STATIC)
    
    @property
    def output_units(self) -> int:
        """
        Set number of the output units of trajectory encoding.
        """
        return self._arg('output_units', 32, argtype=STATIC)
    
    @property
    def generation_num(self) -> int:
        """
        Number of multi-style generation.
        """
        return self._arg('generation_num', 20, argtype=STATIC)
    
    @property
    def view_angle(self) -> float:
        """
        Value of conception view field.
        """
        return self._arg('view_angle', np.pi, argtype=STATIC)
