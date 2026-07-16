"""Generate a 2-channel angular 'CD disc' for the holographic foil.

Like disc-cd (smooth conic sections) but stored in TWO channels so the colour
can sweep continuously all the way around instead of mirroring:

  R = 0.5 + 0.5*cos(k*theta)   <- channel 1
  G = 0.5 + 0.5*sin(k*theta)   <- channel 2
  B = 0 (unused)
  A = disc mask

Together (R,G) encode a rotating unit vector, so the shader recovers the true
angle k*theta via atan2 and drives the palette -> a seamless rainbow that
repeats k times around the disc, converging sharply at the centre.

SPLITS = number of colour repeats around the disc (8 = eight sections).
"""
import numpy as np
from PIL import Image

N = 512
SPLITS = 2            # colour repeats around the disc (2 = two segments per colour)
PHASE_DEG = 0.0       # rotate the pattern

xs = (np.arange(N) + 0.5) / N * 2.0 - 1.0
X, Y = np.meshgrid(xs, xs)
R = np.sqrt(X * X + Y * Y)
inside = R <= 1.0
theta = np.arctan2(Y, X)

k = float(SPLITS)
a = k * theta + np.deg2rad(PHASE_DEG)
Rc = 0.5 + 0.5 * np.cos(a)
Gc = 0.5 + 0.5 * np.sin(a)

r8 = np.clip(np.round(Rc * 255.0), 0, 255).astype(np.uint8)
g8 = np.clip(np.round(Gc * 255.0), 0, 255).astype(np.uint8)
r8 = np.where(inside, r8, 128).astype(np.uint8)   # neutral outside
g8 = np.where(inside, g8, 128).astype(np.uint8)
b8 = np.zeros((N, N), np.uint8)
alpha = np.where(inside, 255, 0).astype(np.uint8)

Image.fromarray(np.dstack([r8, g8, b8, alpha]), "RGBA").save("disc-cd2.png")
print(f"wrote disc-cd2.png ({N}x{N}), {SPLITS} splits")
