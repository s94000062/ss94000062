"""
RT-DETR v4 自動修補腳本
==============================
用法：把此腳本放到 ComfyUI 根目錄（和 main.py 同層），然後執行：

    python apply_fix.py

腳本會自動修補 comfy/model_detection.py、comfy/supported_models.py、
comfy/model_base.py，讓 rt_detr_v4-x-hgnet_fp16.safetensors 能被
正確載入，不再出現 NextDiT.forward() 錯誤。

修補完成後重新啟動 ComfyUI 即可。
"""

import os
import sys
import shutil
import datetime

# ─── 路徑 ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DETECTION_FILE   = os.path.join(BASE, "comfy", "model_detection.py")
SUPPORTED_FILE   = os.path.join(BASE, "comfy", "supported_models.py")
MODEL_BASE_FILE  = os.path.join(BASE, "comfy", "model_base.py")

def backup(path):
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path + f".bak_{ts}"
    shutil.copy2(path, dst)
    print(f"  備份已儲存：{dst}")

def check(path):
    if not os.path.exists(path):
        print(f"[ERROR] 找不到檔案：{path}")
        print("        請確認此腳本放在 ComfyUI 根目錄（和 main.py 同層）")
        sys.exit(1)

# ─── 1. 修補 model_detection.py ──────────────────────────────────────────────
def patch_model_detection():
    check(DETECTION_FILE)
    with open(DETECTION_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    MARKER = "encoder.pan_blocks.1.cv4.conv.weight"
    if MARKER in content:
        print("[model_detection.py] ✓ 已有 RT-DETR v4 偵測，略過。")
        return

    # 插入點：state_dict_keys = list(state_dict.keys()) 的下一行
    INSERT_AFTER = "state_dict_keys = list(state_dict.keys())"
    if INSERT_AFTER not in content:
        print("[model_detection.py] ✗ 找不到插入點，嘗試替代方案…")
        # 嘗試在 def detect_unet_config 函數第一個 if 之前插入
        INSERT_AFTER = "def detect_unet_config("
        if INSERT_AFTER not in content:
            print("[model_detection.py] ✗ 無法自動修補，請手動修改（見下方說明）。")
            _print_manual_detection()
            return

    PATCH = (
        "\n"
        "    if '{}encoder.pan_blocks.1.cv4.conv.weight'.format(key_prefix) in state_dict_keys: # RT-DETR_v4\n"
        "        dit_config = {}\n"
        "        dit_config[\"image_model\"] = \"RT_DETR_v4\"\n"
        "        dit_config[\"enc_h\"] = state_dict['{}encoder.pan_blocks.1.cv4.conv.weight'.format(key_prefix)].shape[0]\n"
        "        return dit_config\n"
    )

    backup(DETECTION_FILE)
    new_content = content.replace(INSERT_AFTER, INSERT_AFTER + PATCH, 1)
    with open(DETECTION_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("[model_detection.py] ✓ RT-DETR v4 偵測已加入。")

def _print_manual_detection():
    print("""
手動修改說明 (model_detection.py)：
找到 detect_unet_config 函數中的這一行：
    state_dict_keys = list(state_dict.keys())

緊接在它後面加上：
    if '{}encoder.pan_blocks.1.cv4.conv.weight'.format(key_prefix) in state_dict_keys:
        dit_config = {}
        dit_config["image_model"] = "RT_DETR_v4"
        dit_config["enc_h"] = state_dict['{}encoder.pan_blocks.1.cv4.conv.weight'.format(key_prefix)].shape[0]
        return dit_config
""")

# ─── 2. 修補 supported_models.py ─────────────────────────────────────────────
def patch_supported_models():
    check(SUPPORTED_FILE)
    with open(SUPPORTED_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    if "RT_DETR_v4" in content:
        print("[supported_models.py] ✓ 已有 RT_DETR_v4，略過。")
        return

    # 在檔案末尾的 models = [...] 之前插入 class 定義，並把 RT_DETR_v4 加入列表
    CLASS_DEF = '''
class RT_DETR_v4(supported_models_base.BASE):
    unet_config = {
        "image_model": "RT_DETR_v4",
    }

    supported_inference_dtypes = [torch.float16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.RT_DETR_v4(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None

'''

    # 找 models = [ 列表定義
    if "models = [" not in content:
        print("[supported_models.py] ✗ 找不到 models = [ 列表，略過。")
        return

    backup(SUPPORTED_FILE)

    # 在 models = [ 前插入 class
    new_content = content.replace("models = [", CLASS_DEF + "models = [", 1)

    # 把 RT_DETR_v4 加進 models 列表（在 ] 前）
    # 找最後一個 ] 結尾的 models 列表
    idx = new_content.rfind("]")
    if idx != -1:
        new_content = new_content[:idx] + ", RT_DETR_v4" + new_content[idx:]

    with open(SUPPORTED_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("[supported_models.py] ✓ RT_DETR_v4 class 與列表項目已加入。")

# ─── 3. 修補 model_base.py ───────────────────────────────────────────────────
def patch_model_base():
    check(MODEL_BASE_FILE)
    with open(MODEL_BASE_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    if "class RT_DETR_v4" in content:
        print("[model_base.py]       ✓ 已有 RT_DETR_v4，略過。")
        return

    # 加在檔案末尾
    CLASS_DEF = '''

class RT_DETR_v4(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        try:
            import comfy.ldm.rt_detr.rtdetr_v4 as _rtdetr
            super().__init__(model_config, model_type, device=device,
                             unet_model=_rtdetr.RTv4)
        except TypeError:
            # Older BaseModel without unet_model parameter
            super().__init__(model_config, model_type, device=device)
            import comfy.ops as _ops
            import comfy.ldm.rt_detr.rtdetr_v4 as _rtdetr
            unet_cfg = dict(model_config.unet_config)
            unet_cfg.pop("image_model", None)
            self.diffusion_model = _rtdetr.RTv4(
                **unet_cfg, device=device,
                operations=_ops.disable_weight_init
            )
'''

    backup(MODEL_BASE_FILE)
    with open(MODEL_BASE_FILE, "a", encoding="utf-8") as f:
        f.write(CLASS_DEF)
    print("[model_base.py]       ✓ RT_DETR_v4 class 已加入。")

# ─── 4. 順便確認模型檔案是否存在且格式正確 ────────────────────────────────────
def check_model_file():
    print("\n=== 檢查模型檔案 ===")
    candidates = [
        os.path.join(BASE, "models", "diffusion_models",
                     "rt_detr_v4-x-hgnet_fp16.safetensors"),
        os.path.join(BASE, "models", "unet",
                     "rt_detr_v4-x-hgnet_fp16.safetensors"),
    ]
    found = next((p for p in candidates if os.path.exists(p)), None)

    if not found:
        print("✗ 找不到 rt_detr_v4-x-hgnet_fp16.safetensors")
        print("  請從此網址下載並放入 models/diffusion_models/ 資料夾：")
        print("  https://huggingface.co/Comfy-Org/SDPose/resolve/main/"
              "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors")
        return

    print(f"✓ 找到模型：{found}")

    try:
        from safetensors.torch import load_file
        sd = load_file(found)
        KEY = "encoder.pan_blocks.1.cv4.conv.weight"
        if KEY in sd:
            print(f"✓ 模型格式正確（含 RT-DETR v4 特徵鍵）")
        else:
            print(f"✗ 警告：模型中找不到 '{KEY}'")
            print("  前 10 個 key：")
            for k in list(sd.keys())[:10]:
                print(f"    {k}: {sd[k].shape}")
            print("\n  這個模型檔案可能不正確！")
            print("  請重新下載：")
            print("  https://huggingface.co/Comfy-Org/SDPose/resolve/main/"
                  "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors")
    except Exception as e:
        print(f"  (無法讀取模型內容：{e}，略過驗證)")

# ─── 主程式 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RT-DETR v4 修補腳本")
    print("=" * 60)

    print("\n=== 修補 ComfyUI 核心檔案 ===")
    patch_model_detection()
    patch_supported_models()
    patch_model_base()

    check_model_file()

    print("\n" + "=" * 60)
    print("完成！請重新啟動 ComfyUI。")
    print("=" * 60)
