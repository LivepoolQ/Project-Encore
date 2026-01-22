import sys

import qpid
import ranger
import socialality

"""
This repo is compatible with our previous models.
Put their model folders (not the whole repo) into this folder to enable them.

File structures:

(root)
    |___qpid
    |___dataset_original
    |___dataset_configs
    |___dataset_processed
    |___main.py
    |___reverberation         
    |___resonance             
    |___socialCircle          
    |___...
"""


if __name__ == '__main__':
    qpid.entrance(sys.argv)
