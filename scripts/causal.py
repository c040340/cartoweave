import numpy as np
from cartoweave.compute.eval import energy_and_grad_full

L = 12
rng = np.random.default_rng(0)
P = rng.standard_normal((L, 2)).astype(float)
mask = np.ones((L,), bool)
labels = [{"mode": "rect"}] * L
scene = {"WH": np.ones((L, 2)), "frame_size": (1920, 1080)}

cfg0 = {
    "public": {"forces": {"anchor.spring": {"enable": True, "k_local": 1.0}, "ll.disk": {"enable": True, "k_out": 1.0}}}
}
cfgw = {
    "public": {"forces": {"anchor.spring": {"enable": True, "k_local": 2.0}, "ll.disk": {"enable": True, "k_out": 1.0}}}
}

E0, G0, C0 = energy_and_grad_full(P, labels, scene, mask, cfg0)
Ew, Gw, Cw = energy_and_grad_full(P, labels, scene, mask, cfgw)

def norm(d, k):
    import numpy as np
    return np.linalg.norm(d.get(k)) if k in d else None

print("keys base:", sorted(C0.keys()))
print("keys scaled:", sorted(Cw.keys()))
print("‖anchor.spring‖ base, k=", norm(C0, "anchor.spring"), norm(Cw, "anchor.spring"))
print("‖ll.disk‖        base, k=", norm(C0, "ll.disk"), norm(Cw, "ll.disk"))
