import glob, json, os, sys
from scipy.stats import binomtest
def load(root):
    return {os.path.basename(os.path.dirname(p)):json.load(open(p)) for p in glob.glob(f"{root}/*/result.json")}
res=lambda d: bool((d.get("eval") or d).get("resolved"))
to=lambda d: d.get("wall_seconds",0)>=595 or d.get("opencode_rc")==124
runs={n:load(r) for n,r in (a.split("=") for a in sys.argv[1:])}
for n,x in runs.items():
    t=[k for k in x if to(x[k])]
    print(f"{n}: n={len(x)} resolved={sum(res(d) for d in x.values())} timeouts={len(t)} (resolved {sum(res(x[k]) for k in t)}) non-timeout resolved {sum(res(x[k]) for k in x if k not in t)}/{len(x)-len(t)}")
ns=list(runs)
for i in range(len(ns)):
    for j in range(i+1,len(ns)):
        a,b=runs[ns[i]],runs[ns[j]]; ks=set(a)&set(b)
        ao=sum(res(a[k]) and not res(b[k]) for k in ks); bo=sum(res(b[k]) and not res(a[k]) for k in ks); both=sum(res(a[k]) and res(b[k]) for k in ks)
        print(f"{ns[i]} vs {ns[j]}: both={both} {ns[i]}_only={ao} {ns[j]}_only={bo} McNemar p={binomtest(bo,ao+bo,0.5).pvalue:.4f}" if ao+bo else "identical")
