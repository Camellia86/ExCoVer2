import os
import json
import base64
import time
from openai import OpenAI

def build_concise_prompt_c(similar_intent_rules):
    """Construct prompt strictly according to specified format"""
    # Process existing confusing intent determination rules table
    similar_intent_display = similar_intent_rules

    prompt = f"""
You need to adjust the confusing intent determination rules table provided to you, retain the core logic and examples that you think are reasonable, and adjust the overall content to a degree suitable for your understanding and application, so that it can help you distinguish confusing intents rather than affect your judgment.

Output Format Requirements:
Strictly output in the following format:

Confusing Intent Discrimination Rules:
[Intent A vs Intent B]: Determination rule content
...

Task Input:

• Confusing intent discrimination rules table to be updated:
{similar_intent_display}


"""

    return prompt
