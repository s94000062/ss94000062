"""
Fix for: ValueError: too many values to unpack (expected 4)
Location: comfy/ldm/lumina/model.py — embed_all()

Root cause:
  The line `B, C, H, W = x.shape` assumes a 4-D tensor (batch, channels,
  height, width), but some Lumina2 latent inputs arrive as 5-D tensors
  (batch, channels, temporal, height, width), causing the unpack to fail.

Fix:
  Detect the 5-D case and flatten the temporal dimension into the batch
  dimension before proceeding with patch embedding, then restore bsz so
  the rest of the function is unaffected.

Usage:
  Run this script from your ComfyUI root directory:
      python fix_lumina_ksampler.py
  It will patch comfy/ldm/lumina/model.py in place (a .bak backup is kept).
"""

import os
import shutil
import sys

TARGET = os.path.join("comfy", "ldm", "lumina", "model.py")

OLD = "    B, C, H, W = x.shape\n    x = self.x_embedder("
NEW = (
    "    if x.dim() == 5:\n"
    "        _B5, _C5, _T5, H, W = x.shape\n"
    "        x = x.reshape(_B5 * _T5, _C5, H, W)\n"
    "        B, C = _B5 * _T5, _C5\n"
    "    else:\n"
    "        B, C, H, W = x.shape\n"
    "    x = self.x_embedder("
)


def main():
    if not os.path.isfile(TARGET):
        print(f"ERROR: {TARGET} not found. Run this script from your ComfyUI root.")
        sys.exit(1)

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if OLD not in src:
        if "if x.dim() == 5:" in src:
            print("Patch already applied — nothing to do.")
        else:
            print("ERROR: Expected code pattern not found. The file may already be patched or the version differs.")
        sys.exit(0)

    patched = src.replace(OLD, NEW, 1)

    backup = TARGET + ".bak"
    shutil.copy2(TARGET, backup)
    print(f"Backup saved to {backup}")

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"Patch applied successfully to {TARGET}")


if __name__ == "__main__":
    main()
