"""Convert the checked-out firmware header into portable NumPy arrays."""
import argparse
from pathlib import Path
import numpy as np
from baseline.weights import load_original_header

def main():
    p=argparse.ArgumentParser(); p.add_argument('header'); p.add_argument('--output',default='artifacts/original_weights.npz'); a=p.parse_args()
    conv,fc=load_original_header(a.header)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output, conv=np.asarray(conv,dtype=np.int8).reshape(16,1,4,4), fc=np.asarray(fc,dtype=np.int8).reshape(10,16,16,16))
    print(f'wrote {a.output}: conv=(16,1,4,4), fc=(10,16,16,16), ternary int8')
if __name__=='__main__': main()
