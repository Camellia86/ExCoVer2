import os
import json
import base64
import time
from openai import OpenAI

def build_concise_prompt_l(context, sticker_text, similar_intent_rules=""): # 需要补充细粒度特征标签表和相似意图判定规则表
    """严格按照指定格式构建prompt"""
    # 处理上下文格式（将制表符转换为更可读的格式）
    if context and context.strip():
        context_display = ", ".join([line.strip() for line in context.split('\t') if line.strip()])
    else:
        context_display = "无"

    # 处理表情包文本
    sticker_text_display = sticker_text if sticker_text and sticker_text.strip() else "无"

    # 处理相似意图判定规则表
    similar_intent_display = similar_intent_rules if similar_intent_rules and similar_intent_rules.strip() else "暂无"

    # 严格按照指定格式构建prompt
    prompt = f"""
## Place the SP-CoT prompt template
{similar_intent_display}

Output Format (Only one intent can be output.):
Intent: [Number + Name]  
Sentiment: [Number + Name]  
Inference Chain:[Briefly describe which fine-grained features you used to infer which latent intents. What was the chat scenario? Finally, how did you determine sentiment and intent?]

- context：{context_display}
- sticker：See the uploaded sticker image
- sticker-text：{sticker_text_display}"""

    return prompt
