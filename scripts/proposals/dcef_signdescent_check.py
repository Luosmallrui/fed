"""Standard EF vs virtual-start EF when local training is sign descent
(a proxy for AdamW with optimizer state reset every round, as in the RoBERTa runs).
"""
import numpy as np, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from dcef_synthetic_check import top_k
# Fresh-Adam-like local training: tau steps of sign descent with lr (Adam's first steps ~ lr*sign(g)).
def local(s, c, lr, tau):
    x = s.copy()
    for _ in range(tau):
        x = x - lr*np.sign(x - c)
    return x
def run(mode, rho, lr=0.01, tau=8, m=64, r=8, N=10, het=1.0, rounds=600, seed=0):
    rng=np.random.default_rng(seed)
    cbar=rng.standard_normal((m,r)); D=[het*rng.standard_normal((m,r)) for _ in range(N)]
    mu=sum(D)/N; C=[cbar+d-mu for d in D]; p=np.full(N,1/N)
    k=max(1,int(round(rho*m*r))); B=np.zeros((m,r)); ec=[np.zeros((m,r)) for _ in C]; es=np.zeros((m,r))
    errs=[]
    for t in range(rounds):
        q=[]
        for i,ci in enumerate(C):
            if mode=="dense":
                u=local(B,ci,lr,tau)-B
            elif mode=="ef":
                u=local(B,ci,lr,tau)-B+ec[i]
            elif mode=="dcef":
                u=local(B+ec[i],ci,lr,tau)-B
            if mode=="dense": q.append(u); continue
            qi=top_k(u,k); ec[i]=u-qi; q.append(qi)
        g=sum(pi*qi for pi,qi in zip(p,q))
        if mode=="dense": B=B+g
        else:
            # Adam-like local steps are not a linear contraction; server keeps plain EF here
            v=g+es; d=top_k(v,k); es=v-d; B=B+d
        errs.append(np.linalg.norm(B-sum(pi*ci for pi,ci in zip(p,C)))**2/np.linalg.norm(cbar)**2)
    errs=np.array(errs); tail=errs[-200:]
    return tail.mean(), tail.max()
print("sign-descent (fresh-Adam-like) local training, tau=8, lr=0.01; tail mean / tail max of relative error")
for het in (0.0,1.0):
    print(f"heterogeneity={het}")
    print(f"   dense: {run('dense',1.0,het=het)[0]:.5f}")
    for rho in (0.03,0.06,0.1,0.2):
        a=run('ef',rho,het=het); b=run('dcef',rho,het=het)
        print(f"   rho={rho:4.2f} (avg delay d/K={1/rho:5.1f})  standard EF: mean={a[0]:.5f} max={a[1]:.5f}   virtual-start EF: mean={b[0]:.5f} max={b[1]:.5f}")
