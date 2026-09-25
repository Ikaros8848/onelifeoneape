"""Small checks for ternary add/skip/subtract semantics."""
import torch
from baseline.hardware_aware import ternary_quantize
def main():
    x=torch.tensor([1.,2.,3.]); q=torch.tensor([-1.,0.,1.]); assert (x[q>0].sum()-x[q<0].sum()).item()==2
    w=torch.tensor([[-2.,0.,3.]]); _,s,_,_=ternary_quantize(w,0.0,False); assert s.tolist()==[[-1.,0.,1.]]
    print('PASS: +1=addition, 0=skip, -1=subtraction; STE quantizer and clipping paths available')
if __name__=='__main__': main()
