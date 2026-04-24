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
    if not input_key or input_key.strip() == "" or "输入你的" in input_key or "从Kaggle" in input_key or "此处填写" in input_key:
        return os.environ.get("GEMINI_API_KEY", "")
    return input_key

# ==========================================
# 节点 1：Gemini 脚本解析器 (3.1 Pro Preview 旗舰版)
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
        real_api_key = get_secure_api_key(gemini_api_key)
        if not real_api_key:
            error_data = [{"zh": "⚠️ 错误：未找到 Gemini API Key！", "en": "Missing API Key"}]
            return (error_data, error_data[0]["zh"])

        # 升级为最新的 3.1 Pro Preview 模型
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={real_api_key}"
        headers = {'Content-Type': 'application/json'}
        
        sys_prompt = """你是一个专业的AI图像提示词工程师和分镜画师。请阅读用户的中文故事脚本，并将其拆分为连续的视觉分镜头。
对于每个镜头，你需要提供两部分内容：
1. 中文画面描述：包括主体、动作、环境、光影、摄像机角度和艺术风格。（用于人工审核修改）
2. 英文提示词：根据中文描述翻译出最适合图像模型理解的英文Prompt。
你必须严格输出一个 JSON 数组格式，不要包含任何 Markdown 标记。格式范例：
[
  {"zh": "镜头1：一只橘猫坐在砖瓦屋顶上...", "en": "Shot 1: An orange cat sitting on a brick roof..."},
  {"zh": "镜头2：...", "en": "Shot 2: ..."}
]"""
        
        data = {
            "contents": [{"parts": [{"text": sys_prompt + "\n\n用户脚本:\n" + script_text}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        
        try:
            print("▶ 正在调用 Gemini 3.1 Pro 解析脚本...")
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            
            text_result = res_json['candidates'][0]['content']['parts'][0]['text']
            prompt_list_dicts = json.loads(text_result)
            text_view = "\n".join([item["zh"] for item in prompt_list_dicts])
            return (prompt_list_dicts, text_view)
            
        except Exception as e:
            print(f"❌ [Gemini 解析错误]: {e}")
            error_data = [{"zh": f"解析失败: {e}", "en": "Error occurred"}]
            return (error_data, error_data[0]["zh"])


# ==========================================
# 节点 2：分镜头中文手工编辑器 (3.1 Pro Preview 版)
# ==========================================
class StoryboardEditorCN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["一键全自动 (忽略下方文本)", "使用手工修改的中文文本"],),
                "manual_text": ("STRING", {"multiline": True, "default": "如果你选择了手工模式，请将第一个节点输出的文本粘贴到这里。..."}),
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
                return ([item["en"] for item in system_data_list],)
            return (["Error: 没有接收到自动列表数据。"],)

        if mode == "使用手工修改的中文文本":
            if not manual_text.strip() or "请将第一个节点" in manual_text:
                return (["Error: 手工文本框为空或使用了默认占位符。"],)
            
            real_api_key = get_secure_api_key(gemini_api_key)
            if not real_api_key:
                return (["Error: 未找到 API Key。"],)

            zh_prompts = [p.strip() for p in manual_text.split("\n") if p.strip()]
            print(f"▶ 正在使用 Gemini 3.1 Pro 重新翻译 {len(zh_prompts)} 个手工镜头...")
            
            # 升级为最新的 3.1 Pro Preview 模型
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={real_api_key}"
            headers = {'Content-Type': 'application/json'}
            sys_prompt = "You are a translation bot. Translate each line of Chinese into a high-quality English image generation prompt. Return ONLY a JSON array of strings. Example: [\"prompt 1\", \"prompt 2\"]"
            
            data = {
                "contents": [{"parts": [{"text": sys_prompt + "\n\n" + "\n".join(zh_prompts)}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            }
            
            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                res_json = response.json()
                text_result = res_json['candidates'][0]['content']['parts'][0]['text']
                return (json.loads(text_result),)
            except Exception as e:
                print(f"❌ [翻译节点错误]: {e}")
                return (["Error: Translation failed."],)


# ==========================================
# 节点 3：批量出图 (Gemini 原生 Nano Banana 2 版)
# ==========================================
class APIBatchGeneratorCN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompts_list": ("LIST",),
            "gemini_api_key": ("STRING", {"default": "默认从Kaggle_Secrets读取，无需填写"}),
            # 默认指定最新版 Flash Image Preview 模型
            "model_name": ("STRING", {"default": "gemini-3.1-flash-image-preview"}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("生成的图像批次",)
    FUNCTION = "generate_batch"
    CATEGORY = "StoryFlow中文版"

    def generate_batch(self, prompts_list, gemini_api_key, model_name):
        images = []
        real_api_key = get_secure_api_key(gemini_api_key)
        
        if not real_api_key:
            print("❌ 错误：未找到 Gemini API Key！返回黑色占位图。")
            error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            error_img[:, :, :, 0] = 1.0 
            return (torch.cat([error_img] * max(1, len(prompts_list)), dim=0),)

        headers = {'Content-Type': 'application/json'}
        # 原生直连 Google AI Studio 接口
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={real_api_key}"

        print(f"▶ 开始批量原生出图任务，使用模型 [{model_name}]，共计 {len(prompts_list)} 张...")

        for i, prompt in enumerate(prompts_list):
            if "Error" in prompt:
                print(f"  ⚠️ 跳过错误提示词...")
                error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                error_img[:, :, :, 0] = 1.0 
                images.append(error_img)
                continue

            print(f"  > 正在生成第 {i+1} 个镜头 (原生API): {prompt[:40]}...")
            
            # 使用标准的 Gemini multimodal payload
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                res_json = response.json()
                
                # 遍历层级结构，安全提取 Base64 图像数据 (inlineData)
                parts = res_json.get('candidates', [{}])[0].get('content', {}).get('parts', [])
                b64_img = None
                for part in parts:
                    if 'inlineData' in part:
                        b64_img = part['inlineData'].get('data')
                        break
                
                if not b64_img:
                    raise ValueError("API 返回成功，但结构中未包含图像 (inlineData)")

                image_data = base64.b64decode(b64_img)
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                images.append(img_tensor)
                print(f"  ✅ 第 {i+1} 张生成成功！")
                
            except Exception as e:
                print(f"  ❌ [图像生成失败]: {e}")
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"  ❌ API 返回详情: {response.text}")
                error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                error_img[:, :, :, 0] = 1.0 
                images.append(error_img)

        if not images:
             batch_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        else:
             batch_tensor = torch.cat(images, dim=0)
             
        print("🎉 批量出图任务完成！")
        return (batch_tensor,)
