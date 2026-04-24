import json
import torch
import numpy as np
from PIL import Image
import io
import requests
import base64
import os

# 辅助函数：安全获取 API Key
def get_secure_api_key(input_key):
    # 如果用户没有输入，或者用的是默认占位符，就去环境变量里找
    if not input_key or input_key.strip() == "" or "输入你的" in input_key or "从Kaggle" in input_key:
        return os.environ.get("GEMINI_API_KEY", "")
    return input_key

# ==========================================
# 节点 1：Gemini 脚本解析器 (中文特化+安全版)
# ==========================================
class GeminiScriptParserCN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "script_text": ("STRING", {"multiline": True, "default": "请输入你的中文故事脚本..."}),
            "gemini_api_key": ("STRING", {"default": "默认从Kaggle_Secrets读取，无需填写"}),
        }}
    
    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("系统数据列表", "人工修改文本")
    FUNCTION = "parse_script"
    CATEGORY = "StoryFlow中文版"

    def parse_script(self, script_text, gemini_api_key):
        # 提取真实 Key
        real_api_key = get_secure_api_key(gemini_api_key)
        
        if not real_api_key:
            error_data = [{"zh": "⚠️ 错误：未找到 Gemini API Key！请在 Kaggle Secrets 配置或直接在节点输入。", "en": "Missing API Key"}]
            return (error_data, error_data[0]["zh"])

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={real_api_key}"
        headers = {'Content-Type': 'application/json'}
        
        sys_prompt = """你是一个专业的AI图像提示词工程师和分镜画师。请阅读用户的中文故事脚本，并将其拆分为连续的视觉分镜头。
对于每个镜头，你需要提供两部分内容：
1. 中文画面描述：包括主体、动作、环境、光影、摄像机角度和艺术风格。（用于人工审核修改）
2. 英文提示词：根据中文描述翻译出最适合图像模型（如Flux/Midjourney）理解的英文Prompt。
你必须严格输出一个 JSON 数组格式，不要包含任何 Markdown 标记。格式范例：
[
  {"zh": "镜头1：一只橘猫坐在砖瓦屋顶上，看着巨大的满月，电影级光影，8k分辨率。", "en": "Shot 1: An orange cat sitting on a brick roof, looking at a huge full moon, cinematic lighting, 8k resolution."},
  {"zh": "镜头2：...", "en": "Shot 2: ..."}
]"""
        
        data = {
            "contents": [{"parts": [{"text": sys_prompt + "\n\n用户脚本:\n" + script_text}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        
        try:
            print("▶ 正在调用 Gemini 解析脚本...")
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            
            text_result = res_json['candidates'][0]['content']['parts'][0]['text']
            prompt_list_dicts = json.loads(text_result)
            
            text_view = "\n".join([item["zh"] for item in prompt_list_dicts])
            return (prompt_list_dicts, text_view)
            
        except Exception as e:
            print(f"❌ [Gemini 解析错误]: {e}")
            error_data = [{"zh": f"解析失败，API无响应或配额耗尽: {e}", "en": "Error occurred"}]
            return (error_data, error_data[0]["zh"])


# ==========================================
# 节点 2：分镜头中文手工编辑器 (安全版)
# ==========================================
class StoryboardEditorCN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["一键全自动 (忽略下方文本)", "使用手工修改的中文文本"],),
                "manual_text": ("STRING", {"multiline": True, "default": "如果你选择了手工模式，请将第一个节点输出的文本粘贴到这里。每行代表一个镜头的画面描述，请随意修改中文内容。我们会自动将其重新翻译为英文送给画图模型..."}),
                "gemini_api_key": ("STRING", {"default": "默认从Kaggle_Secrets读取，无需填写"}),
            },
            "optional": {
                "system_data_list": ("LIST",),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("最终英文提示词列表",)
    FUNCTION = "process_prompts"
    CATEGORY = "StoryFlow中文版"

    def process_prompts(self, mode, manual_text, gemini_api_key, system_data_list=None):
        if mode == "一键全自动 (忽略下方文本)":
            if system_data_list:
                en_prompts = [item["en"] for item in system_data_list]
                return (en_prompts,)
            else:
                return (["Error: 没有接收到自动列表数据。"],)

        if mode == "使用手工修改的中文文本":
            if not manual_text.strip() or "请将第一个节点" in manual_text:
                return (["Error: 手工文本框为空或使用了默认占位符。"],)
            
            real_api_key = get_secure_api_key(gemini_api_key)
            if not real_api_key:
                return (["Error: 未找到 Gemini API Key，无法执行翻译。"],)

            zh_prompts = [p.strip() for p in manual_text.split("\n") if p.strip()]
            print(f"▶ 检测到手工修改了 {len(zh_prompts)} 个镜头，正在重新翻译为英文 Prompt...")
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={real_api_key}"
            headers = {'Content-Type': 'application/json'}
            sys_prompt = "You are a translation bot. The user will give you a list of Chinese image descriptions separated by newlines. Translate each line into a high-quality English image generation prompt. Return ONLY a JSON array of strings. Example: [\"prompt 1\", \"prompt 2\"]"
            
            data = {
                "contents": [{"parts": [{"text": sys_prompt + "\n\n" + "\n".join(zh_prompts)}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            }
            
            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                res_json = response.json()
                text_result = res_json['candidates'][0]['content']['parts'][0]['text']
                final_en_prompts = json.loads(text_result)
                return (final_en_prompts,)
            except Exception as e:
                print(f"❌ [翻译节点错误]: {e}")
                return (["Error: Translation failed."],)


# ==========================================
# 节点 3：批量出图 (API版)
# ==========================================
class APIBatchGeneratorCN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompts_list": ("LIST",),
            "api_url": ("STRING", {"default": "https://api.your-provider.com/v1/images/generations"}),
            "api_key": ("STRING", {"default": "此处填写你的图像大模型_API_KEY"}),
            "model_name": ("STRING", {"default": "gemini-3-flash-image"}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("生成的图像批次",)
    FUNCTION = "generate_batch"
    CATEGORY = "StoryFlow中文版"

    def generate_batch(self, prompts_list, api_url, api_key, model_name):
        images = []
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print(f"▶ 开始批量出图任务，共计 {len(prompts_list)} 张...")

        for i, prompt in enumerate(prompts_list):
            # 防止空列表导致越界
            if "Error" in prompt:
                print(f"  ⚠️ 跳过错误提示词: {prompt}")
                error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                error_img[:, :, :, 0] = 1.0 
                images.append(error_img)
                continue

            print(f"  > 正在生成第 {i+1} 个镜头 (英文Prompt): {prompt[:40]}...")
            
            payload = {
                "prompt": prompt,
                "model": model_name,
                "response_format": "b64_json"
            }
            
            try:
                if "your-provider" in api_url:
                    print(f"  ⚠️ 未配置真实的图像 API URL，返回黑色占位图。")
                    images.append(torch.zeros((1, 512, 512, 3), dtype=torch.float32))
                    continue

                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                res_json = response.json()
                
                b64_img = res_json['data'][0]['b64_json'] 
                image_data = base64.b64decode(b64_img)
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                
                img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                images.append(img_tensor)
                print(f"  ✅ 第 {i+1} 张生成成功！")
                
            except Exception as e:
                print(f"  ❌ [图像生成失败]: {e}")
                error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                error_img[:, :, :, 0] = 1.0 
                images.append(error_img)

        if not images:
             batch_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        else:
             batch_tensor = torch.cat(images, dim=0)
             
        print("🎉 批量任务完成！")
        return (batch_tensor,)
