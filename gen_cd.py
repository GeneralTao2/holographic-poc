"""Generate a smooth 1-channel 'CD disc' segment for the holographic foil.

The tone varies with ANGLE (conic gradient) using a sinusoid, so the light/
dark bands blend into each other with no sharp edges anywhere. SECTIONS sets
how many light/dark lobes go around the disc (4 = CD-like quadrants). The
pattern amplitude eases in from the centre so there is no singular point.
"""
import numpy as np
from PIL import Image

N = 512
SECTIONS = 4          # number of light/dark lobes around the disc
PHASE_DEG = 0.0       # rotate the pattern
CENTER_EASE = 0.0     # radius over which lobes fade in (0 = full contrast to centre)

xs = (np.arange(N) + 0.5) / N * 2.0 - 1.0
X, Y = np.meshgrid(xs, xs)
R = np.sqrt(X * X + Y * Y)
theta = np.arctan2(Y, X)

inside = R <= 1.0

# smooth angular wave: SECTIONS lobes, no discontinuity (sinusoid wraps cleanly)
cycles = SECTIONS / 2.0
if CENTER_EASE > 1e-6:
    amp = np.clip(R / CENTER_EASE, 0.0, 1.0)      # ease pattern in from centre
    amp = amp * amp * (3.0 - 2.0 * amp)           # smoothstep
else:
    amp = np.ones_like(R)                         # full contrast to the centre
wave = np.cos(cycles * theta + np.deg2rad(PHASE_DEG))
val = 0.5 + 0.5 * amp * wave                      # 0..1, perfectly smooth

g = np.clip(np.round(val * 255.0), 0, 255).astype(np.uint8)
alpha = np.where(inside, 255, 0).astype(np.uint8)
g = np.where(inside, g, 0).astype(np.uint8)

Image.fromarray(np.dstack([g, g, g, alpha]), "RGBA").save("disc-cd.png")
print(f"wrote disc-cd.png ({N}x{N}), {SECTIONS} sections")
