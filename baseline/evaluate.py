"""Metrics and operation/data-movement accounting without third-party deps."""
def confusion_matrix(y, p, n=10):
    m=[[0]*n for _ in range(n)]
    for a,b in zip(y,p): m[a][b]+=1
    return m
def metrics(y,p,n=10):
    cm=confusion_matrix(y,p,n); prec=[]; rec=[]; f1=[]
    for i in range(n):
        tp=cm[i][i]; fp=sum(cm[r][i] for r in range(n))-tp; fn=sum(cm[i])-tp
        pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
        prec.append(pr); rec.append(re); f1.append(2*pr*re/(pr+re) if pr+re else 0)
    return {'accuracy':sum(a==b for a,b in zip(y,p))/len(y) if y else 0,'macro_precision':sum(prec)/n,'macro_recall':sum(rec)/n,'macro_f1':sum(f1)/n,'confusion_matrix':cm}
def operation_estimate(num_maps=16, k=4, out=61*61):
    terms=num_maps*k*k*out
    return {'convolution_terms':terms,'multiplications':0,'additions':terms,'subtractions':0,'comparisons':num_maps*(out//2),'approximate_MACs':terms}

def data_movement_estimate(input_hw=64, maps=16, conv_hw=61, pool_hw=30, classes=10, bits=2):
    """Analytical element counts and bit volumes, not hardware traffic."""
    elems={'input':input_hw*input_hw,'weights':maps*4*4,'intermediate':maps*conv_hw*conv_hw,'output':classes}
    return {'elements':elems,'bits':{k:v*bits for k,v in elems.items()}}
