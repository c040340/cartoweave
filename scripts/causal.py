import numpy as np
from cartoweave.compute.eval import energy_and_grad_full

L=12
rng=np.random.default_rng(0)
P=rng.standard_normal((L,2)).astype(float)
mask=np.ones((L,),bool)
scene={'labels':[{'mode':'rect'}]*L,'WH':np.ones((L,2)),'frame_size':(1920,1080)}

cfg0={'terms':{'anchor':{'spring':{'k':10.0}}, 'll':{'disk':{'k':1.0}}}}
cfgw={'terms':{'anchor':{'spring':{'k':10.0}}, 'll':{'disk':{'k':1.0}}}, 'solver':{'internals':{'weights':{'anchor.spring':2.0}}}}

E0,G0,C0,_ = energy_and_grad_full(P, scene, mask, cfg0)
Ew,Gw,Cw,_ = energy_and_grad_full(P, scene, mask, cfgw)

def norm(d,k):
    import numpy as np
    return np.linalg.norm(d.get(k)) if k in d else None

print("keys base:", sorted(C0.keys()))
print("keys wght:", sorted(Cw.keys()))
print("‖anchor.spring‖ base, w=", norm(C0, "anchor.spring"), norm(Cw, "anchor.spring"))
print("‖ll.disk‖        base, w=", norm(C0, "ll.disk"), norm(Cw, "ll.disk"))