"""Generate smooth disc base images for the holographic foil effect.

Outputs (512x512, disc inscribed so it fills the frame edge-to-edge):
  disc-normal.png  RG = surface normal (nx,ny), B = nz, A = disc mask.
                   Use for normal-map lighting -> directional highlights.
  disc-dome.png    Grayscale hemisphere height (1 channel) + alpha.
                   Drop-in for the current luminance->palette shader.

The shape is a spherical cap (hemisphere): perfectly smooth, radially
symmetric, zero angular structure -> eliminates the "cross" artifacts.
A small edge rounding keeps the rim from going fully tangential.
"""
import numpy as np
from PIL import Image

N = 512
# pixel-center coordinates in [-1, 1]
xs = (np.arange(N) + 0.5) / N * 2.0 - 1.0
X, Y = np.meshgrid(xs, xs)
R = np.sqrt(X * X + Y * Y)

inside = R <= 1.0

# --- hemisphere height / normal ---
# z of a unit sphere cap; clamp so sqrt stays valid
z = np.sqrt(np.clip(1.0 - np.minimum(R, 1.0) ** 2, 0.0, 1.0))

# surface normal of the sphere cap is simply (x, y, z) (already unit length)
nx, ny, nz = X.copy(), Y.copy(), z.copy()
# normalize defensively
length = np.sqrt(nx * nx + ny * ny + nz * nz)
length[length == 0] = 1.0
nx, ny, nz = nx / length, ny / length, nz / length

# outside the disc: flat normal (0,0,1) so filtering never pulls in garbage
nx = np.where(inside, nx, 0.0)
ny = np.where(inside, ny, 0.0)
nz = np.where(inside, nz, 1.0)

alpha = np.where(inside, 255, 0).astype(np.uint8)


def enc(v):
    return np.clip(np.round((v * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)


normal_rgba = np.dstack([enc(nx), enc(ny), enc(nz), alpha])
Image.fromarray(normal_rgba, "RGBA").save("disc-normal.png")

# --- grayscale dome (height as luminance) ---
# use a slightly eased dome so the whole 0..1 range is spanned smoothly
height = np.where(inside, z, 0.0)
g = np.clip(np.round(height * 255.0), 0, 255).astype(np.uint8)
dome_rgba = np.dstack([g, g, g, alpha])
Image.fromarray(dome_rgba, "RGBA").save("disc-dome.png")

print("wrote disc-normal.png and disc-dome.png (512x512)")
