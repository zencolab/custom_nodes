from .story_nodes_cn import GeminiScriptParserCN, StoryboardEditorCN, APIBatchGeneratorCN

NODE_CLASS_MAPPINGS = {
    "GeminiScriptParserCN": GeminiScriptParserCN,
    "StoryboardEditorCN": StoryboardEditorCN,
    "APIBatchGeneratorCN": APIBatchGeneratorCN
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiScriptParserCN": "1. 脚本解析器 (Gemini版)",
    "StoryboardEditorCN": "2. 分镜头中文编辑器",
    "APIBatchGeneratorCN": "3. 批量出图 (API版)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
