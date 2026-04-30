"""
Fix: TypeError: NextDiT.forward() missing 2 required positional arguments: 'context' and 'num_tokens'

Root cause
----------
ComfyUI-aki v3 is based on an older ComfyUI snapshot that is missing the RT-DETR v4
entries added later in master across THREE files:

  1. comfy/model_detection.py  – detect_unet_config() has no RT_DETR_v4 key check,
     so rt_detr_v4-x-hgnet_fp16.safetensors falls through to a wrong model type
     (Lumina2 / NextDiT) whose forward() requires 'context' and 'num_tokens'.

  2. comfy/supported_models.py – RT_DETR_v4 config class missing from models list,
     so even if detection succeeds the loader cannot instantiate the right wrapper.

  3. comfy/model_base.py       – RT_DETR_v4 model class missing, so the supported
     model config has nothing to delegate to.

This custom node patches all three at startup time.

Installation
------------
Copy this entire fix_rt_detr_detection/ folder into:
    <ComfyUI root>/custom_nodes/

Restart ComfyUI.  You should see "[fix_rt_detr] All patches applied." in the log.
No other files need to be changed.
"""

import torch
import logging

_LOG = "[fix_rt_detr]"

# ─────────────────────────────────────────────────────────────────────────────
# 0. Verify the RTv4 model class is present (it lives in comfy/ldm/rt_detr/).
#    nodes_rtdetr.py already imports from this module, so if this import fails
#    the whole RTDETR_detect node would be broken beyond this fix.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from comfy.ldm.rt_detr.rtdetr_v4 import RTv4 as _RTv4
    print(f"{_LOG} RTv4 class found.")
except ImportError as exc:
    print(f"{_LOG} FATAL – RTv4 not found in comfy.ldm.rt_detr.rtdetr_v4: {exc}")
    print(f"{_LOG} Your ComfyUI installation is missing the RT-DETR v4 model "
          "implementation entirely.  Update ComfyUI to a version that includes "
          "comfy/ldm/rt_detr/rtdetr_v4.py.")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    # Do not raise – let ComfyUI keep loading other nodes.
    _RTv4 = None


if _RTv4 is not None:

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Patch comfy/model_base.py – add RT_DETR_v4 model class if absent.
    # ─────────────────────────────────────────────────────────────────────────
    import comfy.model_base as _mb

    if not hasattr(_mb, "RT_DETR_v4"):
        try:
            from comfy.model_base import BaseModel, ModelType

            class _RT_DETR_v4_ModelBase(BaseModel):
                def __init__(self, model_config, model_type=ModelType.FLOW,
                             device=None):
                    # 'unet_model' parameter was added in a later ComfyUI version.
                    # Try the modern signature first, fall back for older builds.
                    try:
                        super().__init__(model_config, model_type,
                                        device=device, unet_model=_RTv4)
                    except TypeError:
                        super().__init__(model_config, model_type, device=device)
                        import comfy.ops as _ops
                        unet_cfg = dict(model_config.unet_config)
                        unet_cfg.pop("image_model", None)
                        ops = _ops.disable_weight_init
                        self.diffusion_model = _RTv4(
                            **unet_cfg, device=device, operations=ops
                        )

            _mb.RT_DETR_v4 = _RT_DETR_v4_ModelBase
            print(f"{_LOG} RT_DETR_v4 added to comfy.model_base.")
        except Exception as exc:
            print(f"{_LOG} WARNING – could not add RT_DETR_v4 to model_base: {exc}")
    else:
        print(f"{_LOG} RT_DETR_v4 already in comfy.model_base.")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Patch comfy/supported_models.py – register RT_DETR_v4 config class
    #    in the models list if it is absent.
    # ─────────────────────────────────────────────────────────────────────────
    import comfy.supported_models as _sm
    import comfy.supported_models_base as _smb

    _already_registered = any(
        getattr(m, "unet_config", {}).get("image_model") == "RT_DETR_v4"
        for m in getattr(_sm, "models", [])
    )

    if not _already_registered:
        try:
            class _RT_DETR_v4_Config(_smb.BASE):
                unet_config = {"image_model": "RT_DETR_v4"}
                supported_inference_dtypes = [torch.float16, torch.float32]

                def get_model(self, state_dict, prefix="", device=None):
                    return _mb.RT_DETR_v4(self, device=device)

                def clip_target(self, state_dict={}):
                    return None

            _sm.models.append(_RT_DETR_v4_Config)
            print(f"{_LOG} RT_DETR_v4 config registered in supported_models.models.")
        except Exception as exc:
            print(f"{_LOG} WARNING – could not register RT_DETR_v4 config: {exc}")
    else:
        print(f"{_LOG} RT_DETR_v4 already registered in supported_models.models.")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Patch comfy/model_detection.detect_unet_config – inject RT-DETR v4
    #    key check BEFORE any other detection in the function body.
    #
    #    Detection key: encoder.pan_blocks.1.cv4.conv.weight
    #    (the HGNetv2 PAN block convolution that is unique to RTv4)
    # ─────────────────────────────────────────────────────────────────────────
    import comfy.model_detection as _md

    _RTDETR_KEY = "encoder.pan_blocks.1.cv4.conv.weight"
    _orig_detect = _md.detect_unet_config

    def _patched_detect(*args, **kwargs):
        # Robust arg extraction – handle any signature variant.
        state_dict = args[0] if args else kwargs.get("state_dict", {})
        key_prefix  = args[1] if len(args) > 1 else kwargs.get("key_prefix", "")

        for pfx in (key_prefix, ""):
            probe = f"{pfx}{_RTDETR_KEY}"
            if probe in state_dict:
                print(f"{_LOG} Detected RT_DETR_v4 via key '{probe}'.")
                return {
                    "image_model": "RT_DETR_v4",
                    "enc_h": state_dict[probe].shape[0],
                }

        return _orig_detect(*args, **kwargs)

    _md.detect_unet_config = _patched_detect
    print(f"{_LOG} comfy.model_detection.detect_unet_config patched.")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Belt-and-suspenders: also patch model_config_from_unet so that even
    #    if the detection above is bypassed (e.g. the caller caches a reference
    #    to the original function), RT-DETR loading still succeeds.
    # ─────────────────────────────────────────────────────────────────────────
    _orig_from_unet = getattr(_md, "model_config_from_unet", None)

    if _orig_from_unet is not None:
        def _patched_from_unet(*args, **kwargs):
            result = _orig_from_unet(*args, **kwargs)
            if result is not None:
                return result

            # Original returned None – check if this is an RT-DETR model.
            state_dict = args[0] if args else kwargs.get("state_dict", {})
            key_prefix  = args[1] if len(args) > 1 else kwargs.get("unet_key_prefix", "")
            for pfx in (key_prefix, ""):
                probe = f"{pfx}{_RTDETR_KEY}"
                if probe in state_dict:
                    print(f"{_LOG} model_config_from_unet fallback: "
                          f"building RT_DETR_v4 config from key '{probe}'.")
                    enc_h = state_dict[probe].shape[0]
                    unet_cfg = {"image_model": "RT_DETR_v4", "enc_h": enc_h}
                    for m in getattr(_sm, "models", []):
                        if getattr(m, "unet_config", {}).get("image_model") == "RT_DETR_v4":
                            return m(unet_cfg)
            return None

        _md.model_config_from_unet = _patched_from_unet
        print(f"{_LOG} comfy.model_detection.model_config_from_unet patched.")

    print(f"{_LOG} All patches applied successfully.")


# ComfyUI requires these even for nodes that register no UI nodes.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
