import json
import torch
import numpy as np
from PIL import Image
import io
import os

from google import genai
from google.genai import types

# ==========================================
# 罪魁祸首修复：终极智能密钥拦截器
# ==========================================
def get_secure_api_key(input_key):
    # 只要输入框里包含任何中文字符（不管填了啥），一律去底层拿真实密钥
    if not input_key or input_key.strip() == "" or any('\u4e00' <= c <= '\u9fa5' for c in input_key):
        # 1. 优先尝试 Kaggle Secrets
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("GEMINI_API_KEY")
        except Exception:
            pass
        # 2. 降级尝试系统环境变量
        return os.environ.get("GEMINI_API_KEY", "")
    
    return input_key.strip()

def get_gcp_client(api_key_input):
    real_api_key = get_secure_api_key(api_key_input)
    if not real_api_key:
        print("[StoryFlow] ❌ 致命错误：未能获取到真实的 API Key！")
        return None
    try:
        # 接入 300 刀企业级通道
        return genai.Client(vertexai=True, api_key=real_api_key)
    except Exception as e:
        print(f"[StoryFlow] SDK Client init failed: {e}")
        return None

def clean_json_text(text):
    text = text.strip()
    triple_ticks = chr(96) * 3
    lines = text.split('\n')
    if lines and lines[0].strip().startswith(triple_ticks): lines = lines[1:]
    if lines and lines[-1].strip().startswith(triple_ticks): lines = lines[:-1]
    cleaned = '\n'.join(lines).strip()
    if cleaned.startswith("json\n"): cleaned = cleaned[5:]
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
            error_data = [{"zh": "Error: 找不到 API Key", "en": "Missing API Key"}]
            return (error_data, error_data[0]["zh"])
        
        sys_prompt = "你是一个专业的AI图像提示词工程师和分镜画师。请阅读用户的中文故事脚本，并将其拆分为连续的视觉分镜头。\n对于每个镜头，你需要提供两部分内容：\n1. 中文画面描述：包括主体、动作、环境、光影、摄像机角度和艺术风格。（用于人工审核修改）\n2. 英文提示词：根据中文描述翻译出最适合图像模型理解的英文Prompt。\n你必须严格输出一个 JSON 数组格式，不要包含任何 Markdown 标记。格式范例：\n[\n  {\"zh\": \"镜头1：一只橘猫坐在砖瓦屋顶上...\", \"en\": \"Shot 1: An orange cat sitting on a brick roof...\"},\n  {\"zh\": \"镜头2：...\", \"en\": \"Shot 2: ...\"}\n]"
        try:
            print("[StoryFlow] 正在调用 GCP (Gemini 3.1 Pro) 解析脚本...")
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=f"{sys_prompt}\n\n用户脚本:\n{script_text}"
            )
            clean_text = clean_json_text(response.text)
            prompt_list_dicts = json.loads(clean_text)
            text_view = "\n".join([item["zh"] for item in prompt_list_dicts])
            print("[StoryFlow] ✅ 脚本解析成功！")
            return (prompt_list_dicts, text_view)
        except Exception as e:
            print(f"[StoryFlow] ❌ 解析失败: {e}")
            error_data = [{"zh": f"Error: {e}", "en": "Error occurred"}]
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
                "manual_text": ("STRING", {"multiline": True, "default": ""}),
                "gemini_api_key": ("STRING", {"default": "默认从系统环境变量读取，无需填写"}),
            },
            "optional": {"system_data_list": ("LIST",)}
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("最终英文提示词列表",)
    FUNCTION = "process_prompts"
    CATEGORY = "StoryFlow中文版 (企业级通道)"

    def process_prompts(self, mode, manual_text, gemini_api_key, system_data_list=None):
        if mode == "一键全自动 (忽略下方文本)":
            if system_data_list and "Error" not in system_data_list[0].get("en", ""):
                return ([item["en"] for item in system_data_list],)
            return (["Error: 没有收到自动解析的数据。"],)

        if not manual_text.strip(): return (["Error: 手工文本框为空。"],)
        
        client = get_gcp_client(gemini_api_key)
        if not client: return (["Error: API Key missing."],)

        zh_prompts = [p.strip() for p in manual_text.split("\n") if p.strip()]
        sys_prompt = "You are a translation bot. Translate each line of Chinese into a high-quality English image generation prompt. Return ONLY a JSON array of strings. Example: [\"prompt 1\", \"prompt 2\"]"
        try:
            print(f"[StoryFlow] 正在重译 {len(zh_prompts)} 个手工修改的镜头...")
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=f"{sys_prompt}\n\n" + "\n".join(zh_prompts)
            )
            return (json.loads(clean_json_text(response.text)),)
        except Exception:
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
        error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        error_img[:, :, :, 0] = 1.0 
        
        client = get_gcp_client(gemini_api_key)
        images = []
        if not client: return (torch.cat([error_img] * max(1, len(prompts_list)), dim=0),)

        print(f"[StoryFlow] 开始批量出图任务，使用模型: {model_name}，共计 {len(prompts_list)} 张。")
        img_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9", output_mime_type="image/png")
        )

        for i, prompt in enumerate(prompts_list):
            if "Error" in prompt:
                images.append(error_img)
                continue
            print(f"  [>] 正在生成第 {i+1} 张图...")
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=img_config
                )
                img_bytes = response.candidates[0].content.parts[0].inline_data.data
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0))
                print(f"  [+] 第 {i+1} 张图生成成功！")
            except Exception as e:
                print(f"  [x] 第 {i+1} 张图生成失败: {e}")
                images.append(error_img)

        print("[StoryFlow] 🎉 全部出图任务完成！")
        return (torch.cat(images, dim=0) if images else error_img,)
