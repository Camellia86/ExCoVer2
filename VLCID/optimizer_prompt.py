import os
import json
import base64
import time
from openai import OpenAI

def build_concise_prompt_o(error_samples, similar_intent_rules):
    """
    Build optimizer prompt in batch

    Args:
        error_samples: List of failed samples, each element is a dictionary:
            {
                'context': '...',
                'sticker_text': '...',
                'true_intent': '...',
                'true_sentiment': '...',
                'learner_response': '...'
            }
        fine_grained_features: Existing fine-grained feature table for sentiment correction
        similar_intent_rules: Existing confusing intent determination rules table
    """
    # Process existing confusing intent determination rules table
    similar_intent_display = similar_intent_rules

    # Build batch sample display text
    samples_display = ""
    for idx, sample in enumerate(error_samples, 1):
        context = sample.get('context', '')
        sticker_text = sample.get('sticker_text', '')
        true_intent = sample.get('true_intent', '')
        true_sentiment = sample.get('true_sentiment', '')
        learner_response = sample.get('learner_response', '')

        # Process context format
        if context and context.strip():
            context_display = ", ".join([line.strip() for line in context.split('\t') if line.strip()])
        else:
            context_display = "None"

        # Process sticker text
        sticker_text_display = sticker_text if sticker_text and sticker_text.strip() else "None"

        samples_display += f"""
[Sample {idx}]
• Context: {context_display}
• Text in sticker: {sticker_text_display}
• Correct intent and sentiment labels: Intent={true_intent}, Sentiment={true_sentiment}
• Learner's response: {learner_response}
"""

    # Strictly build prompt according to specified format
    prompt = f"""
Role and Task Definition

You are a social media semantic recognition optimization expert. Your core task is to analyze the reasoning errors made by the Learner when processing "chat context + sticker", trace the root causes of errors, extract confusing intent determination rules, and dynamically update the existing knowledge base.

Note: The sentiment in this task refers to sentiment, which only includes Neutral, Positive, and Negative. Latent intents belong to an open set, can be empty, and are not limited to predefined intent labels.

Stage 1: Confusing Intent Root Cause Diagnosis (internal thinking process)
Logical Priority — For this sample, is the incorrect intent also an auxiliary intent? Did the Learner mistakenly judge the "auxiliary intent" as the "core intent"?
Intent Similarity — Are the intent predicted by the Learner and the correct intent similar intents?
Other Cases — Did the Learner get the perspective wrong from the start?

Stage 2: Cross-sample Universal Feature Extraction
From the following {len(error_samples)} failed samples, extract intent determination rules from each sample, then check whether these rules **do not contradict other samples**. Prioritize patterns that can cover multiple samples rather than special cases for individual samples.

Stage 3: Knowledge Base Update Criteria
Confusing Intent Determination Rules:
For the two intents confused by the Learner, extract clear identification criteria.
Primary-Secondary Logic Determination: If two intents are compatible, specify the determination method for primary and secondary. For example: "This is my friend John; he is really good at playing football", there are "introduce" intent and "praise" intent, prioritize the action-type intent "introduce" as primary.
Similar Intent Determination: If two intents are similar in meaning, pay attention to pointing out the key latent intent, fine-grained features, or method to distinguish them.
Other Cases: If you think this pair of confusing intents does not belong to the above two types, please point out the key fine-grained features, latent intents, or methods that the learner ignored or misunderstood according to the learner's reasoning chain.
Format: [Intent A vs Intent B - Determination Rule]

Stage 4: Rule Maintenance (Deduplication and Merging)
Coverage Upgrade: If there are rules of the same type in the old rules, and the old rule is similar in meaning to the new rule, just supplement the new identification logic in the new rule (it's also possible that the old and new meanings are the same, and no new logic needs to be supplemented). If the old rule fails in the current case, delete the logic in the old rule that contradicts the current case, supplement the logic of the new rule, and ensure that the rules are concise and refined, but do not lose key details, especially the fine-grained features and key latent intents that can distinguish different intents.

Output Format Requirements:
Strictly output in the following format:

Confusing Intent Discrimination Rules:
[Intent A vs Intent B]: Determination rule content
...

Task Input:

[Batch Failed Sample Analysis]
{samples_display}

[Existing Knowledge Base]

• Confusing intent determination rules table to be updated:
{similar_intent_display}

"""

    return prompt
