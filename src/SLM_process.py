import numpy as np
from matplotlib import pyplot as plt

from scipy.io import savemat
import argparse
import hdf5storage

argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input", help="input filename", required=True, nargs='+')


args = argParser.parse_args()
for name in args.input:
    output_filename = (name).split(".npy")[0] 
    print(f"Input file: {name}")
    print(f"Output file: {output_filename}")

    data = np.load(name)
    data = data.astype(np.uint8)
    zero_padded = np.zeros((1920,1152, data.shape[0]), dtype=np.uint8)
    zero_padded[384:384+1152,:,:] = data.T
    mdic = {"turb": zero_padded}

    # savemat(output_filename, mdic)
    hdf5storage.savemat(output_filename, mdic, format=7.3, matlab_compatible=True, compress=False)