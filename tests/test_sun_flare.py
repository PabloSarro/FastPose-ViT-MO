"""FastRandomSunFlare (src/bop_dataset.py) gives bit-identical images to A.RandomSunFlare.

- The two properties of albucore's uint8 add_weighted that the shortcut relies on.
- The transform alone, on random images (float32 as in the pipeline, and uint8), several sizes
  and sun radii, with the same seed (so the same random flare parameters).
- The whole "domain_gap" pipeline with the fast and the original transform.
"""

import albumentations as A
import numpy as np
import pytest
from albucore import add_weighted

from src.bop_dataset import FastRandomSunFlare, pixel_transform


def _images(n, h, w, seed, dtype):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        if k % 2:  # smooth image: large uniform areas, where rounding differences would show
            yy, xx = np.mgrid[0:h, 0:w]
            img = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), xx * 0 + 128 * (k % 3)], -1)
            img = img.astype(np.uint8)
        else:
            img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        out.append(img.astype(np.float32) / 255.0 if dtype == np.float32 else img)
    return out


def test_add_weighted_identity_and_layout_independence():
    v = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    for a in np.linspace(0, 1, 501):
        assert np.array_equal(add_weighted(v, float(a), v, float(1 - a)), v)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = float(rng.random())
        x = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        y = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        full = add_weighted(x, a, y, 1 - a)
        part = add_weighted(np.ascontiguousarray(x[10:50, 3:77]), a, np.ascontiguousarray(y[10:50, 3:77]), 1 - a)
        assert np.array_equal(full[10:50, 3:77], part)


@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
@pytest.mark.parametrize("shape,src_radius", [((384, 384), 400), ((240, 320), 400), ((96, 64), 55)])
def test_fast_sun_flare_is_bit_identical(dtype, shape, src_radius):
    for k, img in enumerate(_images(12, *shape, seed=shape[0] * 1000 + shape[1], dtype=dtype)):
        kw = dict(flare_roi=(0, 0, 1, 1), src_radius=src_radius, p=1.0)
        ref = A.Compose([A.RandomSunFlare(**kw)], seed=100 + k)(image=img)["image"]
        fast = A.Compose([FastRandomSunFlare(**kw)], seed=100 + k)(image=img)["image"]
        assert ref.dtype == fast.dtype
        assert np.array_equal(ref, fast), f"image {k} differs"


def test_domain_gap_pipeline_unchanged():
    ref_t = pixel_transform("domain_gap", fast_sun_flare=False)
    fast_t = pixel_transform("domain_gap", fast_sun_flare=True)
    ref_t.set_random_seed(7)
    fast_t.set_random_seed(7)
    for img in _images(40, 384, 384, seed=3, dtype=np.float32):
        a, b = ref_t(image=img)["image"], fast_t(image=img)["image"]
        assert a.equal(b)
