"""Generate a 2-channel disc normal map for the holographic foil effect.

Encodes a smooth convex dome as a surface normal:
  R = nx  (tilt left/right)   <- channel 1
  G = ny  (tilt up/down)      <- channel 2
  B = unused (0)              (shader reconstructs nz = sqrt(1 - nx^2 - ny^2))
  A = disc mask

SHAPE controls the bump profile:
  'sphere' -> round pillow (classic hemisphere)
  'dome'   -> flatter top, rounded shoulders (softer 2-tone, more reference-like)
Only R and G carry information; the render's colour comes entirely from the
shader lighting + palette, not from the image.
"""
import numpy as np
from PIL import Image

N = 512
SHAPE = "dome"     # 'sphere' or 'dome'
DOME_P = 2.6       # >2 flattens the top for 'dome' (higher = flatter centre)

xs = (np.arange(N) + 0.5) / N * 2.0 - 1.0
X, Y = np.meshgrid(xs, xs)
R = np.sqrt(X * X + Y * Y)
inside = R <= 1.0
Rc = np.minimum(R, 1.0)

if SHAPE == "sphere":
    # height h(r) = sqrt(1 - r^2); slope dh/dr = -r / sqrt(1 - r^2)
    denom = np.sqrt(np.clip(1.0 - Rc * Rc, 1e-6, 1.0))
    slope = -Rc / denom
else:
    # height h(r) = (1 - r^p)^(1/2)-ish flattened dome; derive slope numerically
    h = (1.0 - Rc ** DOME_P)
    h = np.clip(h, 0.0, 1.0) ** 0.5
    # radial slope via analytic derivative of h(r)= (1 - r^p)^0.5
    # dh/dr = 0.5 (1 - r^p)^(-0.5) * (-p r^(p-1))
    base = np.clip(1.0 - Rc ** DOME_P, 1e-6, 1.0)
    slope = 0.5 * base ** (-0.5) * (-DOME_P * Rc ** (DOME_P - 1.0))

# unit radial direction
eps = 1e-6
ux = np.where(R > eps, X / np.maximum(R, eps), 0.0)
uy = np.where(R > eps, Y / np.maximum(R, eps), 0.0)

# surface normal of a height field z=h(r): N = normalize(-dh/dr * (ux,uy), 1)
nx = -slope * ux
ny = -slope * uy
nz = np.ones_like(nx)
inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
nx, ny, nz = nx * inv, ny * inv, nz * inv

# flat normal outside the disc so filtering never drags in garbage
nx = np.where(inside, nx, 0.0)
ny = np.where(inside, ny, 0.0)


def enc(v):
    return np.clip(np.round((v * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)


alpha = np.where(inside, 255, 0).astype(np.uint8)
zeros = np.zeros((N, N), np.uint8)
img = np.dstack([enc(nx), enc(ny), zeros, alpha])
Image.fromarray(img, "RGBA").save("disc-normal.png")
print(f"wrote disc-normal.png ({N}x{N}) shape={SHAPE}")
