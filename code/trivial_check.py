"""S01: graph-free models have infinite certified radius, so certified accuracy only means something if it beats them.
Compares on validation: ego-only GAM (W1=0), plain, robust; and reports how much the robust model still uses the graph."""
import json, numpy as np, torch, torch.nn.functional as F, gam, cert
BUD=[1,2,3,5]; out={}
for ds,K,th,lams in [("Texas",2,1.0,[1e-4]),("Wisconsin",1,1.0,[1e-4]),("Cornell",2,1.0,[1e-3]),("Cora",2,4.0,[0.1]),("Actor",2,2.0,[1e-2]),("minesweeper",1,1.0,[1e-4]),("roman-empire",4,1.0,[1e-2])]:
    g,ms=gam.load(ds); g=g.to("cuda"); ms=[m.to("cuda") for m in ms]; c=int(g.y.max())+1
    tr,va,_=(m[:,0] for m in ms); m=cert.NeighbourGAM(g,c,K=K,theta=th).cuda(); lam=lams[0]
    nodes=va.nonzero().flatten().cpu().numpy(); nodes=nodes[:1500]; yv=g.y[nodes].cpu().numpy()
    def ev():
        p,r=cert.radii(m,nodes,BUD); ok=p==yv
        with torch.no_grad():
            S0,S1=m.scores(); share=float(S1.std()/(S0.std()+S1.std()))
        return dict(clean=round(float(ok.mean()),3),**{f"c{b}":round(float((ok&(r>=b)).mean()),3) for b in BUD},graph_share=round(share,3))
    # ego-only: same features, same solver, no neighbour term.
    best=(-1,None)
    for l in [1e-1,1e-2,1e-3,1e-4,1e-5]:
        W1=m.W1; opt=torch.optim.LBFGS([m.W0,m.b],lr=1.0,max_iter=300,history_size=20,line_search_fn="strong_wolfe")
        with torch.no_grad(): m.W0.zero_(),m.W1.zero_(),m.b.zero_()
        def cl():
            opt.zero_grad(); L=F.cross_entropy((m.Psi@m.W0+m.b)[tr],g.y[tr])+l*(m.W0**2*m.pen.squeeze(0)).sum(); L.backward(); return L
        opt.step(cl)
        with torch.no_grad(): v=float((m().argmax(1)[nodes].cpu().numpy()==yv).mean())
        if v>best[0]: best=(v,l)
    res={"ego_only_clean_equals_all_radii":round(best[0],3)}
    with torch.no_grad(): m.W0.zero_(),m.W1.zero_(),m.b.zero_()
    cert.fit(m,tr,g.y,lam); res["plain"]=ev()
    if tr.sum()<=800:
        for B in (1,2):
            cert.robust_fit(m,tr.nonzero().flatten(),g.y,lam,B=B,steps=200,lr=0.005); res[f"robustB{B}"]=ev()
    out[ds]=res; print(ds,res,flush=True)
json.dump(out,open("../results/trivial_check.json","w"),indent=1)
