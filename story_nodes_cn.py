import json
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
import logging

# ==========================================
# 终极环境修补：强制接管系统标准流与底层日志
# 彻底解决 Kaggle 环境下 'ascii' codec can't encode 报错
# ==========================================
os.environ["PYTHONIOENCODING"] = "utf-8"

# 强制将标准输出和错误流重定向为 UTF-8，忽略无法编码的字符
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 暴力静音 Google SDK 和底层 HTTP 请求库的所有内部日志，防止其乱打印导致崩溃
logging.getLogger("google").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

from google import genai
from google.genai import types

def get_secure_api_key(input_key):
    if not input_key or input_key.strip() == "" or "输入你的" in input_key or "从Kaggle" in input_key or "此处填写" in input_key:
        return os.environ.get("GEMINI_API_KEY", "")
    return input_key

def get_gcp_client(api_key_input):
    real_api_key = get_secure_api_key(api_key_input)
    if not real_api_key:
        return None
    try:
        return genai.Client(vertexai=True, api_key=real_api_key)
    except Exception as e:
        print(f"[StoryFlow] SDK Client init failed: {e}")
        return None

def clean_json_text(text):
    text = text.strip()
    triple_ticks = chr(96) * 3
    lines = text.split('\n')
    if lines and lines[0].strip().startswith(triple_ticks):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith(triple_ticks):
        lines = lines[:-1]
    cleaned = '\n'.join(lines).strip()
    if cleaned.startswith("json\n"):
        cleaned = cleaned[5:]
    return cleaned.strip()

# ==========================================
# 节点 1：文本解析
# ==========================================
class GeminiScriptParserCN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "script_text": ("STRING", {"multiline": True, "default": "请输入你的中文故事脚本..."}),
            "gemini_api_key": ("STRING", {"default": "默认从系统环境变量读取，无需填写"}),
        }}
    
    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("系统数据列表", "人工修改文本")
    FUNCTION = "parse_script"
    CATEGORY = "StoryFlow中文版 (企业级通道)"

    def parse_script(self, script_text, gemini_api_key):
        client = get_gcp_client(gemini_api_key)
        if not client:
            error_data = [{"zh": "Error: Missing API Key", "en": "Missing API Key"}]
            return (error_data, error_data[0]["zh"])
        
        sys_prompt = """你是一个专业的AI图像提示词工程师和分镜画师。请阅读用户的中文故事脚本，并将其拆分为连续的视觉分镜头。
对于每个镜头，你需要提供两部分内容：
1. 中文画面描述：包括主体、动作、环境、光影、摄像机角度和艺术风格。（用于人工审核修改）
2. 英文提示词：根据中文描述翻译出最适合图像模型理解的英文Prompt。
你必须严格输出一个 JSON 数组格式，不要包含任何 Markdown 标记。格式范例：
[
  {"zh": "镜头1：一只橘猫坐在砖瓦屋顶上...", "en": "Shot 1: An orange cat sitting on a brick roof..."},
  {"zh": "镜头2：...", "en": "Shot 2: ..."}
]"""
        try:
            print("[StoryFlow] Calling GCP (Gemini 3.1 Pro) for script parsing...")
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=f"{sys_prompt}\n\n用户脚本:\n{script_text}"
            )
            
            clean_text = clean_json_text(response.text)
            prompt_list_dicts = json.loads(clean_text)
            text_view = "\n".join([item["zh"] for item in prompt_list_dicts])
            return (prompt_list_dicts, text_view)
            
        except Exception as e:
            # 修改了捕获逻辑，避免直接打印未知的异常对象内容引发二次崩溃
            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"[StoryFlow] Parse error caught securely: {error_msg}")
            error_data = [{"zh": f"Parse Error: {error_msg}", "en": "Error occurred"}]
            return (error_data, error_data[0]["zh"])

# ==========================================
# 节点 2：分镜头编辑器
# ==========================================
class StoryboardEditorCN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["一键全自动 (忽略下方文本)", "使用手工修改的中文文本"],),
                "manual_text": ("STRING", {"multiline": True, "default": "如果你选择了手工模式，请将第一个节点输出的文本粘贴到这里。..."}),
                "gemini_api_key": ("STRING", {"default": "默认从系统环境变量读取，无需填写"}),
            },
            "optional": {
                "system_data_list": ("LIST",),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("最终英文提示词列表",)
    FUNCTION = "process_prompts"
    CATEGORY = "StoryFlow中文版 (企业级通道)"

    def process_prompts(self, mode, manual_text, gemini_api_key, system_data_list=None):
        if mode == "一键全自动 (忽略下方文本)":
            if system_data_list:
                return ([item["en"] for item in system_data_list],)
            return (["Error: No auto list data received."],)

        if mode == "使用手工修改的中文文本":
            if not manual_text.strip() or "请将第一个节点" in manual_text:
                return (["Error: Manual text box is empty or default."],)
            
            client = get_gcp_client(gemini_api_key)
            if not client:
                return (["Error: API Key missing or init failed."],)

            zh_prompts = [p.strip() for p in manual_text.split("\n") if p.strip()]
            print(f"[StoryFlow] Re-translating {len(zh_prompts)} manual shots using GCP...")
            
            sys_prompt = "You are a translation bot. Translate each line of Chinese into a high-quality English image generation prompt. Return ONLY a JSON array of strings. Example: [\"prompt 1\", \"prompt 2\"]"
            
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=f"{sys_prompt}\n\n" + "\n".join(zh_prompts)
                )
                clean_text = clean_json_text(response.text)
                return (json.loads(clean_text),)
            except Exception as e:
                print(f"[StoryFlow] Translation error: {e}")
                return (["Error: Translation failed."],)

# ==========================================
# 节点 3：批量出图
# ==========================================
class APIBatchGeneratorCN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompts_list": ("LIST",),
            "gemini_api_key": ("STRING", {"default": "默认从系统环境变量读取，无需填写"}),
            "model_name": ("STRING", {"default": "gemini-3.1-flash-image-preview"}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("生成的图像批次",)
    FUNCTION = "generate_batch"
    CATEGORY = "StoryFlow中文版 (企业级通道)"

    def generate_batch(self, prompts_list, gemini_api_key, model_name):
        client = get_gcp_client(gemini_api_key)
        images = []
        
        if not client:
            print("[StoryFlow] Client init failed! Returning red placeholder.")
            error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            error_img[:, :, :, 0] = 1.0 
            return (torch.cat([error_img] * max(1, len(prompts_list)), dim=0),)

        print(f"[StoryFlow] Batch image generation started: {model_name}, Total: {len(prompts_list)}")

        img_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
                output_mime_type="image/png",
            )
        )

        for i, prompt in enumerate(prompts_list):
            if "Error" in prompt:
                print("  [-] Skipping error prompt...")
                error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                error_img[:, :, :, 0] = 1.0 
                images.append(error_img)
                continue

            print(f"  [>] Generating shot {i+1}...")
            
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=img_config
                )
                
                img_bytes = response.candidates[0].content.parts[0].inline_data.data
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                images.append(img_tensor)
                print(f"  [+] Shot {i+1} success!")
                
            except Exception as e:
                print(f"  [x] Image generation failed for shot {i+1}: {e}")
                error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                error_img[:, :, :, 0] = 1.0 
                images.append(error_img)

        if not images:
             batch_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        else:
             batch_tensor = torch.cat(images, dim=0)
             
        print("[StoryFlow] Batch generation complete!")
        return (batch_tensor,)
