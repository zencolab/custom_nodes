import json
import torch
import numpy as np
from PIL import Image
import io
import requests
import base64

# ==========================================
# 节点 1：Gemini 脚本解析器 (中文特化)
# ==========================================
class GeminiScriptParserCN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "script_text": ("STRING", {"multiline": True, "default": "请输入你的中文故事脚本..."}),
            "gemini_api_key": ("STRING", {"default": "输入你的_GEMINI_API_KEY"}),
        }}
    
    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("系统数据列表", "人工修改文本")
    FUNCTION = "parse_script"
    CATEGORY = "StoryFlow中文版"

    def parse_script(self, script_text, gemini_api_key):
        # 强制要求输出特定格式的 JSON，包含中文描述和英文提示词
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
        headers = {'Content-Type': 'application/json'}
        
        # 纯中文系统提示词
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
            "generationConfig": {"response_mime_type": "application/json"} # 强制 JSON 输出
        }
        
        try:
            print("▶ 正在调用 Gemini 解析脚本...")
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            
            text_result = res_json['candidates'][0]['content']['parts'][0]['text']
            prompt_list_dicts = json.loads(text_result) # 解析为字典列表
            
            # 将中文描述提取出来，拼接成多行文本供用户修改
            text_view = "\n".join([item["zh"] for item in prompt_list_dicts])
            
            return (prompt_list_dicts, text_view)
            
        except Exception as e:
            print(f"❌ [Gemini 节点错误]: {e}")
            error_data = [{"zh": "解析失败，请检查控制台报错", "en": "Error occurred"}]
            return (error_data, error_data[0]["zh"])


# ==========================================
# 节点 2：分镜头中文手工编辑器
# ==========================================
class StoryboardEditorCN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["一键全自动 (忽略下方文本)", "使用手工修改的中文文本"],),
                "manual_text": ("STRING", {"multiline": True, "default": "如果你选择了手工模式，请将第一个节点输出的文本粘贴到这里。每行代表一个镜头的画面描述，请随意修改中文内容。我们会自动将其重新翻译为英文送给画图模型..."}),
                "gemini_api_key": ("STRING", {"default": "翻译修改内容需要API密钥"}),
            },
            "optional": {
                "system_data_list": ("LIST",), # 接收来自节点 1 的原始字典列表
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("最终英文提示词列表",)
    FUNCTION = "process_prompts"
    CATEGORY = "StoryFlow中文版"

    def process_prompts(self, mode, manual_text, gemini_api_key, system_data_list=None):
        if mode == "一键全自动 (忽略下方文本)":
            if system_data_list:
                # 提取英文提示词
                en_prompts = [item["en"] for item in system_data_list]
                return (en_prompts,)
            else:
                return (["Error: 没有接收到自动列表数据。"],)

        # ====== 手工修改模式 ======
        if mode == "使用手工修改的中文文本":
            if not manual_text.strip():
                return (["Error: 手工文本框为空。"],)
            
            # 将多行中文按行拆分
            zh_prompts = [p.strip() for p in manual_text.split("\n") if p.strip()]
            print(f"▶ 检测到手工修改了 {len(zh_prompts)} 个镜头，正在重新翻译为英文 Prompt...")
            
            # 调用 Gemini 将用户修改后的中文翻译为高质量英文提示词
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
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
            "api_key": ("STRING", {"default": "输入你的_图像生成_API_KEY"}),
            "model_name": ("STRING", {"default": "gemini-3-flash-image"}), # 填 Banana 2 或 flux 等模型名
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
            print(f"  > 正在生成第 {i+1} 个镜头 (英文Prompt): {prompt[:40]}...")
            
            # 【重要配置区】：这里的 payload 格式基于 OpenAI 标准图生图接口，
            # 如果你的 Banana 2/Flux API 接口格式不同，请参照服务商文档修改这里。
            payload = {
                "prompt": prompt,
                "model": model_name,
                "response_format": "b64_json" # 要求 API 返回 base64 数据
            }
            
            try:
                if "your-provider" in api_url:
                    print(f"  ⚠️ 未配置真实的 API URL，返回黑色占位图。")
                    images.append(torch.zeros((1, 512, 512, 3), dtype=torch.float32))
                    continue

                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                res_json = response.json()
                
                # 提取 base64 数据 (注意：不同服务商的 JSON 层级可能不同，通常是 data[0].b64_json 或 image_base64)
                b64_img = res_json['data'][0]['b64_json'] 
                image_data = base64.b64decode(b64_img)
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                
                # 转换给 ComfyUI 识别的 Tensor
                img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                images.append(img_tensor)
                print(f"  ✅ 第 {i+1} 张生成成功！")
                
            except Exception as e:
                print(f"  ❌ [图像生成失败]: {e}")
                # 生成失败时返回一张红色的错误图，避免批次崩溃
                error_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                error_img[:, :, :, 0] = 1.0 
                images.append(error_img)

        if not images:
             batch_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        else:
             batch_tensor = torch.cat(images, dim=0) # 将所有单图拼接为一个批次
             
        print("🎉 批量任务完成！")
        return (batch_tensor,)
