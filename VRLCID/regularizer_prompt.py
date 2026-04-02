import os
import json
import base64
import time
from openai import OpenAI

def build_concise_prompt_r(similar_intent_rules):
    """Construct prompt strictly according to specified format"""
    # Process existing confusing intent discrimination rules table
    similar_intent_display = similar_intent_rules

    prompt = f"""
You are a social cognition and logical induction expert. Your responsibility is to review, refine, and optimize the "social media intent recognition knowledge base". You need to ensure that each rule in the knowledge base has universality and is not merely "overfitting" to specific cases.

Note: The sentiment in this task refers to sentiment, which only includes Neutral, Positive, and Negative. Latent intents belong to an open set, can be empty, and are not limited to predefined intent labels.

Core Review Logic (internal thinking guidance)
Rule Preservation: For rules in the original rule set that are not involved in this update, please preserve them. For example, if only [introduce vs praise] rules need to be updated this time, then other rules must be preserved intact.

For rules involved in the update:
Debiasing and Generalization: Review details in discrimination rules, correct inherent biases that the optimizer may have, and generalize proper nouns and events. Bias example: if nonsensical or model-fabricated rule details like "drinking water"-"guessing" appear, delete them. Generalization example: generalize "Zhang San's laughter" to "laughter"; generalize "very good at playing soccer" to "praise", reaching a degree like [puppy image - cute pet image] is sufficient, while specific latent intents must be retained.
Semantic Deduplication: [Thumbs up] and [like gesture] are synonymous and need to be merged into one more accurate entry, such as [Thumbs up].
Rule Convergence: If two discrimination rules have similar logic, merge them into one more comprehensive criterion, but be careful to retain key latent intents, fine-grained features, and specific methods that distinguish different intents, and avoid over-generalization that leads to loss of specific rule details. If the logic is different, retain the one with the least forced logic, and be careful not to change the names of intents and latent intents in the rules.

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
