from cartoweave.api import solve_frame
import numpy as np

scene = dict(
    frame=0,
    frame_size=(1920,1080),
    points=np.array([[100.0,100.0]]),
    lines=np.zeros((0,4)),
    areas=np.zeros((0,6)),
    labels_init=np.array([[120.0,120.0]]),
)

P, info = solve_frame(scene, cfg={}, mode="hybrid")
print("P:", P)
print("info:", info)
