"""S01: does warm-started exact subgradient descent fix certified training? Compares validation only."""
import json, sys, numpy as np, torch, gam, cert
BUD=[1,2,3,5]
out={}
for ds,K,th,lam in [("Texas",2,1.0,1e-4),("Wisconsin",1,1.0,1e-4),("Cora",2,4.0,0.1),("Cornell",2,1.0,1e-3)]:
    g,ms=gam.load(ds); g=g.to("cuda"); ms=[m.to("cuda") for m in ms]; c=int(g.y.max())+1
    tr,va,_=(m[:,0] for m in ms); m=cert.NeighbourGAM(g,c,K=K,theta=th).cuda()
    nodes=va.nonzero().flatten().cpu().numpy(); yv=g.y[va].cpu().numpy()
    def ev():
        p,r=cert.radii(m,nodes,BUD); ok=p==yv
        return dict(clean=round(float(ok.mean()),3),**{f"c{b}":round(float((ok&(r>=b)).mean()),3) for b in BUD})
    cert.fit(m,tr,g.y,lam); res=[("plain",ev())]; 
    for B,lr in [(1,0.02),(1,0.005),(2,0.02),(2,0.005)]:
        best,h=cert.robust_fit(m,tr.nonzero().flatten(),g.y,lam,B=B,steps=200,lr=lr)
        res.append((f"B{B} lr{lr} obj {h[0]:.3f}->{best:.3f}",ev()))
    # Multi-start: a convex objective must reach about the same value from different starts.
    objs=[]
    for s in range(3):
        torch.manual_seed(s)
        for t in (m.W0,m.W1,m.b): torch.nn.init.normal_(t.data,std=0.05)
        b,_=cert.robust_fit(m,tr.nonzero().flatten(),g.y,lam,B=1,steps=600,lr=0.02,warm=False); objs.append(round(b,4))
    res.append(("cold starts obj",objs)); out[ds]=res
    for r in res: print(ds,r,flush=True)
json.dump(out,open("../results/solver_check.json","w"),indent=1)
