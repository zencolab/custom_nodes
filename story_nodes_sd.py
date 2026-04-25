import torch
import numpy as np
from PIL import Image
import io
import os
import time

from google import genai
from google.genai import types

def get_secure_api_key(input_key):
    # 智能拦截器：如果不填或者填了中文，直接去底层拿真密钥
    if not input_key or input_key.strip() == "" or any('\u4e00' <= c <= '\u9fa5' for c in input_key):
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("GEMINI_API_KEY")
        except Exception:
            pass
        return os.environ.get("GEMINI_API_KEY", "")
    return input_key.strip()

def get_gcp_client(api_key_input):
    real_api_key = get_secure_api_key(api_key_input)
    if not real_api_key:
        print("[StoryFlow-SD] ❌ 致命错误：未能获取到真实的 API Key！")
        return None
    try:
        # 接入 300 刀企业级通道
        return genai.Client(vertexai=True, api_key=real_api_key)
    except Exception as e:
        print(f"[StoryFlow-SD] SDK Client init failed: {e}")
        return None

# 将 ComfyUI 的图像 Tensor 转换为 Google API 需要的字节流
def tensor_to_bytes(tensor):
    img_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

# ==========================================
# 节点 1：纯手工提示词编辑器 (极简版)
# ==========================================
class ManualPromptEditorSD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "manual_prompts": ("STRING", {
                    "multiline": True, 
                    "default": "在此处直接输入你的提示词，每行代表一个分镜头...\n(注：Gemini 图像模型原生支持中文，你可以直接输入中文或英文描述)"
                }),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("提示词列表",)
    FUNCTION = "process_prompts"
    CATEGORY = "StoryFlow-SD (参考图控制版)"

    def process_prompts(self, manual_prompts):
        if not manual_prompts.strip(): 
            return (["Error: 提示词框为空，请输入内容"],)

        # 按行分割字符串，自动过滤掉空行，生成列表
        prompts = [p.strip() for p in manual_prompts.split("\n") if p.strip()]
        
        if not prompts:
            return (["Error: 未检测到有效的提示词"],)
            
        return (prompts,)

# ==========================================
# 节点 2：云端批量出图 (支持 10 张参考图注入)
# ==========================================
class APIBatchGeneratorSD:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "prompts_list": ("LIST",),
                "gemini_api_key": ("STRING", {"default": "默认从系统环境变量读取，无需填写"}),
                "model_name": ("STRING", {"default": "gemini-3.1-flash-image-preview"}),
            },
            "optional": {}
        }
        # 动态创建 10 个图像参考输入端口
        for i in range(1, 11):
            inputs["optional"][f"ref_image_{i}"] = ("IMAGE",)
        return inputs
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("生成的图像批次",)
    FUNCTION = "generate_batch"
    CATEGORY = "StoryFlow-SD (参考图控制版)"

    def generate_batch(self, prompts_list, gemini_api_key, model_name, **kwargs):
        client = get_gcp_client(gemini_api_key)
        images = []
        
        # 如果没有拿到密钥，直接返回红图兜底
        if not client: 
            error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            error_img[:, :, :, 0] = 1.0
            return (torch.cat([error_img] * max(1, len(prompts_list)), dim=0),)

        # ---------------------------------------------------------
        # 收集并预处理参考图像
        # ---------------------------------------------------------
        ref_parts = []
        for i in range(1, 11):
            img_tensor = kwargs.get(f"ref_image_{i}")
            if img_tensor is not None:
                try:
                    img_bytes = tensor_to_bytes(img_tensor)
                    ref_parts.append(types.Part.from_bytes(data=img_bytes, mime_type='image/png'))
                    print(f"[StoryFlow-SD] ✅ 成功加载 IP 参考图 {i}")
                except Exception as e:
                    print(f"[StoryFlow-SD] ⚠️ 警告：参考图 {i} 加载失败 - {e}")

        print(f"\n[StoryFlow-SD] 开始出图任务: {model_name}，共 {len(prompts_list)} 张")
        if ref_parts:
            print(f"[StoryFlow-SD] 🔗 已向云端注入 {len(ref_parts)} 张 IP 角色参考图！")

        # 画幅比例设定
        img_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9", output_mime_type="image/png")
        )

        for i, prompt in enumerate(prompts_list):
            if "Error" in prompt:
                print(f"  [-] 第 {i+1} 张：前期报错跳过。")
                images.append(None)
                continue
            
            print(f"  [>] 正在生成第 {i+1} 张图...")
            success = False
            max_retries = 3 
            
            # 核心机制：把参考图数组和当前行的提示词打包在一起发给模型
            payload_contents = ref_parts + [prompt]
            
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model_name, contents=payload_contents, config=img_config
                    )
                    img_bytes = response.candidates[0].content.parts[0].inline_data.data
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0))
                    print(f"  [+] 第 {i+1} 张图生成成功！")
                    success = True
                    break 
                    
                except Exception as e:
                    error_msg = str(e)
                    # 遭遇限流，自动退避等待
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "Quota" in error_msg:
                        wait_time = 15 * (attempt + 1)
                        print(f"  [!] 触发频率限制 (429)。等待 {wait_time} 秒后重试 ({attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"  [x] 生成失败: {e}")
                        break
            
            if not success:
                images.append(None)
            
            # 防封锁冷却
            time.sleep(3)

        # 动态维度匹配修复（防止红图导致拼接崩溃）
        valid_tensors = [img for img in images if img is not None]
        ref_shape = valid_tensors[0].shape if valid_tensors else (1, 512, 512, 3) 
        error_img = torch.zeros(ref_shape, dtype=torch.float32)
        error_img[:, :, :, 0] = 1.0 

        final_images = [img if img is not None else error_img for img in images]
        print("[StoryFlow-SD] 🎉 任务完成！")
        return (torch.cat(final_images, dim=0),)
