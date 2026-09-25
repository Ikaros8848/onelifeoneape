"""Reference CPU implementation of SCAMP-style ternary operations."""
from typing import Sequence

def conv2d_ternary(image, weights, padding=0):
    """Cross-correlation: +1 adds, -1 subtracts, 0 skips multiplication."""
    h,w=len(image),len(image[0]); kh,kw=len(weights),len(weights[0]); out=[]
    for y in range(h-kh+1+2*padding):
        row=[]
        for x in range(w-kw+1+2*padding):
            s=0.0
            for j in range(kh):
                for i in range(kw):
                    yy,xx=y+j-padding,x+i-padding
                    if 0<=yy<h and 0<=xx<w:
                        q=weights[j][i]
                        if q==1: s += image[yy][xx]
                        elif q==-1: s -= image[yy][xx]
                        elif q not in (0,): s += image[yy][xx]*q
            row.append(s)
        out.append(row)
    return out

def maxpool2d(x, size=2, stride=2):
    return [[max(x[y+j][x0+i] for j in range(size) for i in range(size))
             for x0 in range(0,len(x[0])-size+1,stride)]
            for y in range(0,len(x)-size+1,stride)]

def relu(x): return [[max(0.0,v) for v in row] for row in x]

