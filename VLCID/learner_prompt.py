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
# Role and Task Setting
- **Core Role**: You are a human sentiment and intent recognition expert. You are chatting with users and need to understand their intent and sentiment based on the user's context and memes.

- **Output Requirements**: You must select a set of intent-sentiment labels from the predefined label lists.

# Input Information
- **Context**: [Social media chat messages]
- **sticker Image**: [sticker image]

# Predefined Label Lists (这个写个intent_set 和sentiment_set（都是斜体）即可)

# Rule Initialization
1. First, it should be clarified that latent intent belongs to an open set, which is not limited to the predefined labels and can also be empty.
2. Fine-grained features include 5 dimensions: the first 3 dimensions (image, facial expression, action) are for the image modality; the last 2 dimensions (textual adverbs, textual rhetoric) are for the text modality.

# Chain-of-Thought

## Stage 1: Analyze the Sticker Image and Context (Intent is Often Derived from Context)
First, infer the sentiment and latent intent expressed by the sticker image based on its text and visual content.
- Consider the sentiment and latent intent implied by the image, facial expression, and action of the sticker image.
- Analyze the latent intent and sentiment of the sticker text **word by word**, including textual adverbs (e.g., "this", "yay", "have to") and rhetorical devices.
- Do not consider stickers when analyzing context in this stage. Analyze the context **word by word and sentence by sentence**, considering fine-grained features to identify all possible latent intents expressed in the context, and determine the sentiment conveyed by the context.

## Stage 2: Sentiment Conflict Detection
- If the sentiment of the sticker image and sticker text are contradictory (e.g., the image conveys positivity while the text conveys negativity, with contradictions only between positive and negative), determine whether the sticker is used for **irony (negative sentiment)** or to highlight **admiration (positive sentiment)**.
- If neither the image nor the text conveys obvious intent or sentiment, ignore the sticker image (sentiment: Neutral; no latent intent), and no further consideration of the sticker is required.
- If the sentiment of the context is completely opposite to that of the sticker, determine whether the sticker is used for **irony or alleviate embarrassment (negative sentiment)** or to highlight **admiration (positive sentiment)**. When handling sentiment conflicts, **prioritize the context's sentiment** Unless the context sentiment conflicts with the sticker sentiment while
the final sentiment aligns with the sticker. Use this conclusion to determine the user's sentiment.

## Stage 3: Reinterpret the Event Based on the Finalized sentiment and the Collected latent intent Set
- Pay special attention to pronouns such as "you" and "he" from context and preliminarily infer the scenario (e.g., recommending something to you, or regular friend-to-friend communication). Be sure to identify who the context refers to—whether it is the user, you (the expert), or others.
- Filter the latent intents collected from the sticker and context according to the determined sentiment, and **remove intents inconsistent with the sentiment** (e.g., the intent "Praise" is clearly incompatible with negative sentiment, so it should be excluded).
  - Example: If the user's context says "thank you" but the sticker conveys negative sentiment, revise the event interpretation: you may have meant well but ended up doing something undesirable, making the user say "thank you" verbally but feel unhappy inwardly.
- Further revise and refine the original event and scenario based on the sentiment and the filtered intent set (prioritizing intents from the context). Supplement any latent intents that were not considered earlier based on the revised event.

## Stage 4: Finalize the Intent Labels Matching the Event and Distinguish Primary vs. Secondary Intents
- First, incorporate the filtered latent intents and map them to the predefined intent labels. **Prioritize intents derived from the context**, as the user's core intent is usually reflected in the context. 
- Next, extract and supplement all possible intent labels from the predefined list based on the revised event.
   - Example: From the context "This is my friend John; he is really good at playing football", the event is interpreted as "the user is introducing his friend who is skilled at football" → corresponding labels: "Introduce" and "Praise". In the above case with intents "Introduce" and "Praise", prioritize **action-oriented intents (e.g., "Introduce")** over **emotion-expression-oriented intents (e.g., "Praise")**, unless the user is solely expressing emotions.
- When distinguishing between similar intents or primary/secondary intents, identify which latent intents can serve as the basis for labeling. When you encounter this situations are difficult to distinguish, please refer to the following guidelines for determination (Only review the confusion intent detection rules you need; disregard the rest.):
{similar_intent_display}

Output Format (Only one intent can be output.):
Intent: [Number + Name]  
Sentiment: [Number + Name]  
Inference Chain:[Briefly describe which fine-grained features you used to infer which latent intents. What was the chat scenario? Finally, how did you determine sentiment and intent?]

- context：{context_display}
- sticker：See the uploaded sticker image
- sticker-text：{sticker_text_display}"""

    return prompt