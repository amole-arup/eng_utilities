"""
OPS_utilities.py
https://openseespydoc.readthedocs.io/en/latest/src/RCFrameGravity.html

https://openseespydoc.readthedocs.io/en/latest/src/RCFramePushOver.html

"""

import openseespy.opensees as ops

# Create nodes
# ------------

# Set parameters for overall model geometry
width = 360.0
height = 144.0

def create_nodes():
    # Create nodes
    #    tag, X, Y, Z
    ops.node(1, 0.0, 0.0, 0.0)



def build_model(E2K_dict):
    ops.wipe()
    # Create ModelBuilder (with three-dimensions and 6 DOF/node)
    ops.model('basic', '-ndm', 3, '-ndf', 6)




def run_analysis():
    pass


def main():
    # Later I will set it to either:
    # 1. Run an example
    # 2. Prompt to run a model.
    print('This is a work in progress')


if __name__ == "__main__":
    main()
    # ops.wipe()
