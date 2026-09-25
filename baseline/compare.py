"""Layer-wise comparison helpers for reference vs SCAMP-style outputs."""
def summarize(name, a, b=None):
    vals=[v for row in a for v in row] if a and isinstance(a[0],list) else list(a)
    out={'layer':name,'shape':[len(a),len(a[0])] if a and isinstance(a[0],list) else [len(a)],'min':min(vals,default=0),'max':max(vals,default=0),'mean':sum(vals)/len(vals) if vals else 0}
    if b is not None:
        rhs=[v for row in b for v in row] if b and isinstance(b[0],list) else list(b)
        errs=[abs(x-y) for x,y in zip(vals,rhs)]
        out.update(max_abs_error=max(errs,default=0),mean_abs_error=sum(errs)/len(errs) if errs else 0)
    return out

