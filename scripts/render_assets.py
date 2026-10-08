#!/usr/bin/env python3
"""Render the app icon (SVG, PNG, ICO, ICNS) and copy it into the web UI.

    python scripts/render_assets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
WEB = ROOT / "triggercollide" / "web"

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#1d2650"/><stop offset="1" stop-color="#0b1020"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" rx="224" fill="url(#bg)"/>
  <circle cx="400" cy="540" r="230" fill="none" stroke="#33d6c2" stroke-width="64"/>
  <circle cx="624" cy="540" r="230" fill="none" stroke="#ff5d73" stroke-width="64"/>
  <path d="M512 339 A230 230 0 0 1 512 741 A230 230 0 0 1 512 339 Z" fill="#ffb547"/>
  <g stroke="#ffb547" stroke-width="34" stroke-linecap="round">
    <line x1="512" y1="150" x2="512" y2="236"/>
    <line x1="372" y1="196" x2="420" y2="262"/>
    <line x1="652" y1="196" x2="604" y2="262"/>
  </g>
</svg>
"""


def render(size: int = 1024) -> Image.Image:
    scale = 4
    s = size * scale
    k = s / 1024
    base = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    grad = Image.new("RGBA", (s, s))
    gd = ImageDraw.Draw(grad)
    for y in range(s):
        t = y / s
        c = tuple(int(a + (b - a) * t) for a, b in zip((29, 38, 80), (11, 16, 32), strict=True))
        gd.line([(0, y), (s, y)], fill=c + (255,))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=int(224 * k), fill=255)
    base.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(base)
    r = 230 * k
    left = (400 * k, 540 * k)
    right = (624 * k, 540 * k)
    # filled lens = intersection of two discs
    a = Image.new("L", (s, s), 0)
    ImageDraw.Draw(a).ellipse([left[0] - r, left[1] - r, left[0] + r, left[1] + r], fill=255)
    b = Image.new("L", (s, s), 0)
    ImageDraw.Draw(b).ellipse([right[0] - r, right[1] - r, right[0] + r, right[1] + r], fill=255)
    lens = ImageChops.multiply(a, b)
    glow = Image.new("RGBA", (s, s), (255, 181, 71, 0))
    glow.putalpha(lens.filter(ImageFilter.GaussianBlur(40 * k)).point(lambda v: int(v * 0.6)))
    base = Image.alpha_composite(base, glow)
    d = ImageDraw.Draw(base)
    width = int(64 * k)
    d.ellipse([left[0] - r, left[1] - r, left[0] + r, left[1] + r], outline=(51, 214, 194, 255), width=width)
    d.ellipse([right[0] - r, right[1] - r, right[0] + r, right[1] + r], outline=(255, 93, 115, 255), width=width)
    amber = Image.new("RGBA", (s, s), (255, 181, 71, 255))
    base.paste(amber, (0, 0), lens)
    for x1, y1, x2, y2 in ((512, 150, 512, 236), (372, 196, 420, 262), (652, 196, 604, 262)):
        d.line([(x1 * k, y1 * k), (x2 * k, y2 * k)], fill=(255, 181, 71, 255), width=int(34 * k))
        for x, y in ((x1, y1), (x2, y2)):
            rr = 17 * k
            d.ellipse([x * k - rr, y * k - rr, x * k + rr, y * k + rr], fill=(255, 181, 71, 255))
    return base.resize((size, size), Image.LANCZOS)


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "icon.svg").write_text(SVG, encoding="utf-8")
    big = render(1024)
    big.save(ASSETS / "icon-1024.png")
    big.resize((512, 512), Image.LANCZOS).save(ASSETS / "icon-512.png")
    big.resize((512, 512), Image.LANCZOS).save(ASSETS / "icon.png")
    big.save(ASSETS / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    big.save(ASSETS / "icon.icns")
    shutil.copy(ASSETS / "icon.svg", WEB / "icon.svg")
    big.resize((256, 256), Image.LANCZOS).save(WEB / "icon.png")
    big.save(WEB / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    print("Icons written to assets/ and triggercollide/web/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
