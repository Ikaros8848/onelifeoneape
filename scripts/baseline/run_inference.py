import argparse, json
from baseline.model import ScampCNN
from baseline.preprocessing import mnist_to_64, save_pgm
from baseline.evaluate import operation_estimate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sample-pgm'); ap.add_argument('--output',default='artifacts/reports/baseline_report.json'); a=ap.parse_args()
    image=[[0.0]*28 for _ in range(28)]; image[10][10]=1.0
    x=mnist_to_64(image); 
    if a.sample_pgm: save_pgm(x,a.sample_pgm)
    result=ScampCNN().forward(x)
    report={'note':'software simulation / analytical estimate; no SCAMP-5 hardware','input_shape':[64,64],'conv_kernel':[4,4],'filters':16,'classes':10,'prediction':result['pred'],'operations':operation_estimate()}
    with open(a.output,'w') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
