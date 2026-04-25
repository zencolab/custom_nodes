# 导入原有的自动分镜流程节点
from .story_nodes_cn import GeminiScriptParserCN, StoryboardEditorCN, APIBatchGeneratorCN
# 导入新增的手工参考图流程节点
from .story_nodes_sd import ManualPromptEditorSD, APIBatchGeneratorSD

NODE_CLASS_MAPPINGS = {
    # CN 自动版节点
    "GeminiScriptParserCN": GeminiScriptParserCN,
    "StoryboardEditorCN": StoryboardEditorCN,
    "APIBatchGeneratorCN": APIBatchGeneratorCN,
    
    # SD 手工设定版节点
    "ManualPromptEditorSD": ManualPromptEditorSD,
    "APIBatchGeneratorSD": APIBatchGeneratorSD
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # CN 自动版菜单名称
    "GeminiScriptParserCN": "1. 脚本解析器 (Gemini版)",
    "StoryboardEditorCN": "2. 分镜头中文编辑器",
    "APIBatchGeneratorCN": "3. 批量出图 (API版)",
    
    # SD 手工设定版菜单名称
    "ManualPromptEditorSD": "1. 纯手工提示词编辑器 (SD版)",
    "APIBatchGeneratorSD": "2. 批量出图+参考图 (SD版)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
