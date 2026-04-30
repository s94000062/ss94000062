"""
Fix: TypeError: NextDiT.forward() missing 2 required positional arguments: 'context' and 'num_tokens'

Root cause
----------
ComfyUI's model_detection.detect_unet_config() does not contain (or runs too late)
the RT-DETR v4 detection check.  When UNETLoader loads
rt_detr_v4-x-hgnet_fp16.safetensors, the detection falls through to the
Lumina2/NextDiT branch, so model.model.diffusion_model ends up being a NextDiT
object instead of an RTv4 object.

nodes_rtdetr.py line 40 then calls:
    results.extend(model.model.diffusion_model(image_in, (W, H)))

That invokes NextDiT.forward(x, timesteps, ...) where 'context' and 'num_tokens'
are missing, producing the TypeError.

Fix
---
This custom node patches detect_unet_config() at startup to inject the
RT-DETR v4 key check BEFORE any other detection runs, so the model is always
identified and loaded as RT_DETR_v4 (diffusion_model = RTv4 instance).

Installation
------------
Copy this entire fix_rt_detr_detection/ folder into:
    <ComfyUI root>/custom_nodes/

Restart ComfyUI.  No other files need to be changed.
"""

import inspect
import comfy.model_detection as _md

# ── capture the original function once ────────────────────────────────────────
_original_detect = _md.detect_unet_config

# The key that uniquely identifies an RT-DETR v4 safetensors file.
# (HGNetv2 backbone encoder PAN block convolution weight)
_RTDETR_KEY = "encoder.pan_blocks.1.cv4.conv.weight"


def _patched_detect_unet_config(state_dict, key_prefix, metadata=None):
    """
    Wrapper around detect_unet_config that adds RT-DETR v4 detection before
    the original logic runs.  Required for ComfyUI forks / older builds that
    are missing this detection in their model_detection.py.
    """
    probe_key = "{}{}".format(key_prefix, _RTDETR_KEY)

    if probe_key in state_dict:
        dit_config = {
            "image_model": "RT_DETR_v4",
            "enc_h": state_dict[probe_key].shape[0],
        }
        return dit_config

    # Also check without the prefix in case the file was saved differently.
    if _RTDETR_KEY in state_dict:
        dit_config = {
            "image_model": "RT_DETR_v4",
            "enc_h": state_dict[_RTDETR_KEY].shape[0],
        }
        return dit_config

    # Delegate everything else to the original implementation.
    sig = inspect.signature(_original_detect)
    if "metadata" in sig.parameters:
        return _original_detect(state_dict, key_prefix, metadata)
    else:
        return _original_detect(state_dict, key_prefix)


# ── apply the patch ────────────────────────────────────────────────────────────
_md.detect_unet_config = _patched_detect_unet_config

print("[fix_rt_detr_detection] RT-DETR v4 model detection patch applied.")

# ComfyUI requires these mappings even for nodes that register no custom nodes.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
