import json
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
import logging
import traceback  # 引入终极追踪核武器

# 暴力修补环境
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.getLogger("google").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

from google import genai
from google.genai import types

def get_secure_api_key(input_key):
    if not input_key or input_key.strip() == "" or "输入你的" in input_key or "从Kaggle" in input_key or "此处填写" in input_key:
        return os.environ.get("GEMINI_API_KEY", "")
    return input_key

def get_gcp_client(api_key_input):
    real_api_key = get_secure_api_key(api_key_input)
    if not real_api_key: return None
    try:
        return genai.Client(vertexai=True, api_key=real_api_key)
    except Exception:
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
# 节点 1：带深层追踪的解析器
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
        # 初始化追踪缓冲池，所有过程全部记录
        trace_log = []
        trace_log.append("=== [深层追踪启动] ===")
        trace_log.append(f"> 宿主系统默认编码: {sys.getdefaultencoding()}")
        
        try:
            trace_log.append("> 1. 初始化 GCP 客户端...")
            client = get_gcp_client(gemini_api_key)
            if not client: raise ValueError("Client 初始化失败")

            trace_log.append("> 2. 组装中文系统提示词...")
            sys_prompt = "你是一个专业的AI图像提示词工程师和分镜画师。请阅读用户的中文故事脚本，并将其拆分为连续的视觉分镜头。\n对于每个镜头，你需要提供两部分内容：\n1. 中文画面描述：包括主体、动作、环境、光影、摄像机角度和艺术风格。（用于人工审核修改）\n2. 英文提示词：根据中文描述翻译出最适合图像模型理解的英文Prompt。\n你必须严格输出一个 JSON 数组格式，不要包含任何 Markdown 标记。格式范例：\n[\n  {\"zh\": \"镜头1：一只橘猫坐在砖瓦屋顶上...\", \"en\": \"Shot 1: An orange cat sitting on a brick roof...\"},\n  {\"zh\": \"镜头2：...\", \"en\": \"Shot 2: ...\"}\n]"
            
            full_contents = f"{sys_prompt}\n\n用户脚本:\n{script_text}"
            trace_log.append(f"> 3. 准备调用 generate_content (字符数: {len(full_contents)})")

            # 💣 炸弹预警：接下来这一步如果崩溃，说明是 SDK 内部处理中文时死了
            trace_log.append("> 4. -> 进入 google-genai SDK 内部通讯...")
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=full_contents
            )
            trace_log.append("> 5. <- 成功从 SDK 返回响应！")
            
            trace_log.append("> 6. 正在清理并解析 JSON 结果...")
            clean_text = clean_json_text(response.text)
            prompt_list_dicts = json.loads(clean_text)
            text_view = "\n".join([item["zh"] for item in prompt_list_dicts])
            
            return (prompt_list_dicts, text_view)
            
        except Exception as e:
            trace_log.append("\n💥 捕捉到致命崩溃！堆栈详情如下：")
            # 完整扒下系统底层的报错堆栈
            error_stack = traceback.format_exc()
            trace_log.append(error_stack)
            
            # 将所有追踪信息强行转换为 ASCII 安全字符串，防止二次崩溃
            safe_log = "\n".join(trace_log).encode('ascii', 'replace').decode('ascii')
            
            error_data = [{"zh": "Error", "en": "Error"}]
            # 把这段长长的报错直接丢给 UI 文本框显示
            return (error_data, safe_log)

# ==========================================
# 节点 2：分镜头编辑器 (简化版防断联)
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
        if system_data_list and "Error" not in system_data_list[0].get("en", ""):
            return ([item["en"] for item in system_data_list],)
        return (["Error"],)

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
        error_img[:, :, :, 0] = 1.0 # 红色警告图
        
        client = get_gcp_client(gemini_api_key)
        images = []
        if not client: return (torch.cat([error_img] * max(1, len(prompts_list)), dim=0),)

        img_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9", output_mime_type="image/png")
        )

        for prompt in prompts_list:
            if "Error" in prompt:
                images.append(error_img)
                continue
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=img_config
                )
                img_bytes = response.candidates[0].content.parts[0].inline_data.data
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0))
            except Exception:
                images.append(error_img)

        return (torch.cat(images, dim=0) if images else error_img,)
