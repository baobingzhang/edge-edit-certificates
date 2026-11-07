"""S01, validation only: clean vs certified frontier of the composite against its two parts."""
import json, numpy as np, torch, torch.nn.functional as F, gam, cert, fallback
BUD=[1,2,3,5,6,10,15]; out={}
for ds,K,th,lam in [("Texas",2,1.0,1e-4),("Wisconsin",1,1.0,1e-4),("Cornell",2,1.0,1e-3),("Cora",2,4.0,0.1),("Actor",2,2.0,1e-2),("minesweeper",1,1.0,1e-4),("roman-empire",4,1.0,1e-2),("amazon-ratings",4,1.0,1e-3)]:
    g,ms=gam.load(ds); g=g.to("cuda"); ms=[m.to("cuda") for m in ms]; c=int(g.y.max())+1
    tr,va,_=(m[:,0] for m in ms); m=cert.NeighbourGAM(g,c,K=K,theta=th).cuda()
    nodes=va.nonzero().flatten().cpu().numpy()[:1500]; yv=g.y[nodes].cpu().numpy()
    best=(-1,None)
    for l in [1e-1,1e-2,1e-3,1e-4,1e-5]:
        opt=torch.optim.LBFGS([m.W0,m.b],lr=1.0,max_iter=300,history_size=20,line_search_fn="strong_wolfe")
        with torch.no_grad(): m.W0.zero_(),m.W1.zero_(),m.b.zero_()
        def cl():
            opt.zero_grad(); L=F.cross_entropy((m.Psi@m.W0+m.b)[tr],g.y[tr])+l*(m.W0**2*m.pen.squeeze(0)).sum(); L.backward(); return L
        opt.step(cl)
        with torch.no_grad(): p=(m.Psi@m.W0+m.b).argmax(1)[nodes].cpu().numpy()
        v=float((p==yv).mean())
        if v>best[0]: best=(v,p)
    pe=best[1]
    with torch.no_grad(): m.W0.zero_(),m.W1.zero_(),m.b.zero_()
    cert.fit(m,tr,g.y,lam); pg,rad=cert.radii(m,nodes,BUD)
    res=dict(ego=round(best[0],3),graph_clean=round(float((pg==yv).mean()),3),agree=round(float((pg==pe).mean()),3))
    for t in (1,2,3,5):
        r=fallback.evaluate(pg,rad,pe,yv,t,[1,2,3,5]); res[f"t{t}"]={k:round(v,3) for k,v in r.items()}
    out[ds]=res; print(ds,res,flush=True)
json.dump(out,open("../results/fallback_check.json","w"),indent=1)
