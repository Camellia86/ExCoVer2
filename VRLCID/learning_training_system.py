import os
import json
import base64
import time
import re
from typing import Dict, Tuple, List, Any, Optional
from openai import OpenAI
from learner_prompt import build_concise_prompt_l
from optimizer_prompt import build_concise_prompt_o
from regularizer_prompt import build_concise_prompt_r
from adapt import build_concise_prompt_c
from a_result3 import extract_labels_from_response

# Initialize client (Learner & Optimizer)
client = OpenAI(
    api_key="Your API_KEY",
    base_url="API_URL",
)

# Initialize Regularizer client (using Gemini-3)
regularizer_client = OpenAI(
    api_key="Your API_KEY",
    base_url="API_URL",
)

# Intent label mapping
INTENT_MAPPING = {
    "Complain": 0, "Praise": 1, "Agree": 2, "Compromise": 3, "Query": 4,
    "Joke": 5, "Oppose": 6, "Inform": 7, "Ask for help": 8, "Greet": 9,
    "Taunt": 10, "Introduce": 11, "Guess": 12, "Leave": 13, "Advise": 14,
    "Flaunt": 15, "Criticize": 16, "Thank": 17, "Comfort": 18, "Apologize": 19
}

INTENT_ID_TO_NAME = {v: k for k, v in INTENT_MAPPING.items()}

# English intent label to target language mapping (legacy)
INTENT_EN_TO_CN = {
    "Complain": "Complain",
    "Praise": "Praise",
    "Agree": "Agree",
    "Compromise": "Compromise",
    "Query": "Query",
    "Joke": "Joke",
    "Oppose": "Oppose",
    "Inform": "Inform",
    "Ask for help": "Ask for help",
    "Greet": "Greet",
    "Taunt": "Taunt",
    "Introduce": "Introduce",
    "Guess": "Guess",
    "Leave": "Leave",
    "Advise": "Advise",
    "Flaunt": "Flaunt",
    "Criticize": "Criticize",
    "Thank": "Thank",
    "Comfort": "Comfort",
    "Apologize": "Apologize"
}

# ID to English label reverse mapping
INTENT_ID_TO_EN = {v: k for k, v in {
    "Complain": 0, "Praise": 1, "Agree": 2, "Compromise": 3, "Query": 4,
    "Joke": 5, "Oppose": 6, "Inform": 7, "Ask for help": 8, "Greet": 9,
    "Taunt": 10, "Introduce": 11, "Guess": 12, "Leave": 13, "Advise": 14,
    "Flaunt": 15, "Criticize": 16, "Thank": 17, "Comfort": 18, "Apologize": 19
}.items()}

SENTIMENT_MAPPING = {
    "Neutral": 0, "Positive": 1, "Negative": 2
}

SENTIMENT_ID_TO_NAME = {v: k for k, v in SENTIMENT_MAPPING.items()}

# Sentiment label ID mapping
SENTIMENT_EN_TO_ID = {
    "Neutral": 0,
    "Positive": 1,
    "Negative": 2
}


class LearningTrainingSystem:
    """Language-based learning training system"""

    def __init__(self, train_json_path: str, base_image_path: str = "", batch_size: int = 1):
        """
        Initialize learning training system

        Args:
            train_json_path: Training data JSON file path
            base_image_path: Base image path
            batch_size: Batch size (default=1 means immediate update, >1 means accumulate batch_size samples before update)
        """
        self.train_json_path = train_json_path
        self.base_image_path = base_image_path

        # Batch processing configuration
        self.batch_size = batch_size
        print(f"[INIT DEBUG] Received batch_size parameter: {batch_size}")
        print(f"[INIT DEBUG] self.batch_size set to: {self.batch_size}")
        self.batch_sample_count = 0  # Current batch sample count
        self.batch_errors = []  # Error samples in current batch
        self.batch_count = 0  # Completed batch count

        # Load training data
        self.train_data = self._load_train_data()

        # Initialize rules table
        self.similar_intent_rules = self._load_similar_intent_rules()

        # Training statistics
        self.training_stats = {
            "total_samples": len(self.train_data),
            "correct_count": 0,
            "error_count": 0,
            "optimizer_calls": 0,
            "regularizer_calls": 0,
            "cleaner_calls": 0
        }

    def _load_train_data(self) -> List[Dict]:
        """Load training data"""
        try:
            with open(self.train_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} training samples")
            return data
        except Exception as e:
            print(f"Failed to load training data: {e}")
            return []

    def _load_similar_intent_rules(self) -> str:
        """Load similar intent determination rules"""
        rules_file = "Confusing Intent Determination Rules.txt"
        try:
            if os.path.exists(rules_file):
                with open(rules_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Failed to load rules table: {e}")
        return ""

    def _save_agent_response_json(self, sample_id: int, agent_name: str, response: str,
                                   item: Dict = None, pred_intent: str = "", pred_sentiment: int = -1,
                                   true_intent: str = "", true_sentiment: int = -1):
        """Save Agent response to complete json file (including sample information)"""
        try:
            # Create agent_responses directory (if not exists)
            os.makedirs("agent_responses", exist_ok=True)

            # Build filename: agent_responses/sample_{id}_{agent}.json
            json_file = f"agent_responses/sample_{sample_id:05d}_{agent_name}.json"

            # Build complete response data (including sample information)
            response_data = {
                "sample_id": sample_id,
                "agent": agent_name,
                "context": item.get('context', '') if item else "",
                "sticker": item.get('sticker', '') if item else "",
                "sticker_text": item.get('sticker_text', '') if item else "",
                "true_intent": true_intent,
                "true_sentiment": true_sentiment,
                "pred_intent": pred_intent if pred_intent else "",
                "pred_sentiment": pred_sentiment if pred_sentiment != -1 else "",
                "agent_response": response,
                "timestamp": time.time()
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)

            print(f"✓ {agent_name} response saved: {json_file}")
            return json_file

        except Exception as e:
            print(f"❌ Failed to save {agent_name} response: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _save_feature_and_rules(self, index: int):
        """Save rules table (periodic checkpoint)"""
        rules_file = f"Confusing Intent Determination Rules_step{index}.txt"

        try:
            # Copy current content from main file as checkpoint
            with open("Confusing Intent Determination Rules.txt", 'r', encoding='utf-8') as f:
                content = f.read()
            with open(rules_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✓ Saved rules table checkpoint for step {index}")
        except Exception as e:
            print(f"Failed to save rules table checkpoint: {e}")

    def local_img_to_url(self, img_path: str) -> str:
        """Convert local image to Base64 URL"""
        try:
            if not os.path.exists(img_path):
                return ""
            ext = os.path.splitext(img_path)[1].lstrip('.')
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/{ext};base64,{b64}"
        except Exception as e:
            print(f"Failed to convert image: {e}")
            return ""

    def call_learner(self, item: Dict) -> Tuple[str, str, int]:
        """
        Call learner to get prediction results

        Returns:
            (model_response, pred_intent, pred_sentiment)
            - pred_intent: English intent label (e.g., 'Praise')
            - pred_sentiment: Sentiment numeric label (0-2)
        """
        try:
            context = item.get('context', '')
            sticker_text = item.get('sticker_text', '')

            # Build prompt
            prompt = build_concise_prompt_l(
                context,
                sticker_text,
                self.similar_intent_rules
            )

            # Build image URL (if exists)
            sticker_id = item.get('sticker', '')
            image_content = []

            if self.base_image_path and sticker_id:
                img_path_png = os.path.join(self.base_image_path, f"{sticker_id}.png")
                img_path_webp = os.path.join(self.base_image_path, f"{sticker_id}.webp")

                img_path = None
                if os.path.exists(img_path_png):
                    img_path = img_path_png
                elif os.path.exists(img_path_webp):
                    img_path = img_path_webp

                if img_path:
                    img_url = self.local_img_to_url(img_path)
                    if img_url:
                        image_content.append({
                            "type": "image_url",
                            "image_url": {"url": img_url}
                        })

            # Add text content
            image_content.append({
                "type": "text",
                "text": prompt
            })

            # Call API
            completion = client.chat.completions.create(
                model="doubao-seed-1-6-vision-250815",
                messages=[
                    {
                        "role": "user",
                        "content": image_content
                    }
                ],
                temperature=0.0
            )

            # Extract response content
            model_response = completion.choices[0].message.content or ""

            # Extract labels (returns ID)
            pred_intent_id, pred_sentiment = extract_labels_from_response(model_response)

            # Convert intent ID to English label
            pred_intent = INTENT_ID_TO_EN.get(pred_intent_id, '')

            return model_response, pred_intent, pred_sentiment

        except Exception as e:
            print(f"Failed to call learner: {e}")
            return "", "", -1

    def call_optimizer(self, error_samples: list):
        """
        Batch call optimizer to update feature and rules tables

        Args:
            error_samples: List of failed samples, each element is a dict:
                {
                    'item': {...},  # Original sample data
                    'model_response': '...',  # Learner's response
                    'pred_intent': '...',  # Learner predicted intent
                    'pred_sentiment': int,  # Learner predicted sentiment
                    'true_intent': '...',  # Correct intent
                    'true_sentiment': int  # Correct sentiment
                }
        """
        if not error_samples:
            print("⚠️  No failed samples, skipping optimizer")
            return ""

        try:
            # Build sample format required by optimizer
            formatted_samples = []
            for sample in error_samples:
                item = sample['item']
                context = item.get('context', '')
                sticker_text = item.get('sticker_text', '')
                true_intent = sample['true_intent']
                true_sentiment = sample['true_sentiment']
                learner_response = sample['model_response']

                # Convert English labels (for prompt)
                true_intent_cn = INTENT_EN_TO_CN.get(true_intent, true_intent)
                true_sentiment_name = SENTIMENT_ID_TO_NAME.get(true_sentiment, f"Sentiment{true_sentiment}")

                formatted_samples.append({
                    'context': context,
                    'sticker_text': sticker_text,
                    'true_intent': true_intent_cn,
                    'true_sentiment': true_sentiment_name,
                    'learner_response': learner_response
                })

            # Build optimizer prompt (batch)
            prompt = build_concise_prompt_o(
                formatted_samples,
                self.similar_intent_rules
            )

            # Call API
            print(f"🔧 Calling Optimizer to process {len(error_samples)} failed samples...")
            completion = client.chat.completions.create(
                model="doubao-seed-1-6-vision-250815",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0
            )

            optimizer_response = completion.choices[0].message.content or ""

            # Note: After Optimizer returns suggestions, the caller in train() will immediately call _update_tables_from_response() to extract content
            # Then Regularizer processes the updated tables
            self.training_stats["optimizer_calls"] += 1

            return optimizer_response

        except Exception as e:
            print(f"❌ Failed to call optimizer: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def call_regularizer(self, is_global: bool = False):
        """Call regularizer to validate and optimize tables

        Args:
            is_global: If True, process global table files; if False, process tables in current memory
        """
        try:
            # If global processing, read tables from file first
            if is_global:
                try:
                    with open("Confusing Intent Determination Rules.txt", 'r', encoding='utf-8') as f:
                        rules_to_process = f.read()
                    print("\n🔄 Global Regularizer: Reading tables from file for processing...")
                except Exception as e:
                    print(f"❌ Failed to read global table file: {e}")
                    return ""
            else:
                # Use tables in memory (may have been updated by Optimizer)
                rules_to_process = self.similar_intent_rules

            # Build prompt (no need for optimizer_response, regularizer directly processes current table state)
            prompt = build_concise_prompt_r(
                rules_to_process
            )

            # Call API (using Gemini-3, enable deep thinking mode low)
            completion = regularizer_client.chat.completions.create(
                model="gemini-3-flash-preview",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_body={
                    "thinking_level": "low",
                    "temperature": 0.0
                }
            )

            regularizer_response = completion.choices[0].message.content or ""

            # Update tables
            if is_global:
                # Global regularizer: extract tables but don't call _save_tables_to_files (avoid redundancy)
                # Handle file saving in outer layer directly
                self._update_tables_from_response(regularizer_response, "regularizer_global")
            else:
                self._update_tables_from_response(regularizer_response, "regularizer")
            self.training_stats["regularizer_calls"] += 1

            return regularizer_response

        except Exception as e:
            print(f"Failed to call regularizer: {e}")
            return ""

    def call_cleaner(self):
        """Call cleaner to further clean tables after regularizer optimization"""
        try:
            # Use tables in memory (already optimized by Regularizer)
            rules_to_process = self.similar_intent_rules

            # Build prompt
            prompt = build_concise_prompt_c(
                rules_to_process
            )

            # Call API (using same client as Optimizer)
            print(f"🧹 Calling Cleaner to further clean tables...")
            completion = client.chat.completions.create(
                model="doubao-seed-1-6-vision-250815",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0
            )

            cleaner_response = completion.choices[0].message.content or ""

            # Extract table content from Cleaner response and update
            if cleaner_response:
                self._update_tables_from_response(cleaner_response, "cleaner")
                self.training_stats["cleaner_calls"] += 1

            return cleaner_response

        except Exception as e:
            print(f"❌ Failed to call cleaner: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _update_tables_from_response(self, response: str, source: str):
        """Extract updated tables from response - based on clear section titles"""
        try:
            print(f"\n🔍 [{source}] Starting to parse response...")

            # Extract Confusing Intent Discrimination Rules section - from "Confusing Intent Discrimination Rules：" to end
            rules_pattern = r"Confusing Intent Discrimination Rules：(.*)"
            rules_match = re.search(rules_pattern, response, re.DOTALL)

            rules_extracted = False
            if rules_match:
                updated_rules = rules_match.group(1).strip()
                if updated_rules:  # Prevent empty update
                    self.similar_intent_rules = updated_rules
                    rules_extracted = True
                    print(f"  ✓ Confusing Intent Discrimination Rules: Match successful ({len(updated_rules)} characters)")
                else:
                    print(f"  ⚠ Confusing Intent Discrimination Rules: Matched but content is empty")
            else:
                print(f"  ✗ Confusing Intent Discrimination Rules: Not found")

            # Save updated tables to txt files
            if rules_extracted:
                # If Optimizer output, save to ablation files and main table files
                if source == "optimizer":
                    self._save_ablation_tables()
                    self._save_tables_to_files()
                elif source == "regularizer_global":
                    self._save_global_regularizer_tables()
                elif source == "cleaner":
                    self._save_cleaner_tables()
                # # Other cases (sample-level regularizer): append to files
                # else:
                #     self._save_tables_to_files()
                print(f"\n✅ [{source}] Tables successfully extracted and saved")
            else:
                print(f"\n⚠️  [{source}] Failed to extract any table content")

        except Exception as e:
            print(f"❌ Failed to update tables ({source}): {e}")
            import traceback
            traceback.print_exc()

    def _save_ablation_tables(self):
        """Save Optimizer extracted tables to ablation files"""
        try:
            ablation_rules_file = "ablation-rules.txt"

            # Save to ablation rules file
            with open(ablation_rules_file, "a", encoding="utf-8") as f:
                f.write(self.similar_intent_rules)
            print(f"    📊 Ablation rules table saved: {ablation_rules_file} ({len(self.similar_intent_rules)} characters)")

        except Exception as e:
            print(f"❌ Failed to save ablation tables to file: {e}")

    def _save_tables_to_files(self):
        """Save current tables to txt files"""
        try:
            rules_file = "Confusing Intent Determination Rules.txt"

            # Save Confusing Intent Determination Rules table
            with open(rules_file, "a", encoding="utf-8") as f:
                f.write(self.similar_intent_rules)
            print(f"    💾 Rules table saved: {rules_file} ({len(self.similar_intent_rules)} characters)")

        except Exception as e:
            print(f"❌ Failed to save tables to file: {e}")
    def _save_global_regularizer_tables(self):
        """Directly overwrite main table file (Global Regularizer output) - no redundancy"""
        try:
            rules_file = "Confusing Intent Determination Rules.txt"
            # Directly overwrite Confusing Intent Determination Rules table
            with open(rules_file, "w", encoding="utf-8") as f:
                f.write(self.similar_intent_rules)
            print(f"    ✅ Rules table overwritten (Global Regularizer): {rules_file}")

        except Exception as e:
            print(f"❌ Failed to save Global Regularizer tables to file: {e}")

    def _save_cleaner_tables(self):
        """Directly overwrite main table file (Cleaner output)"""
        try:
            rules_file = "Confusing Intent Determination Rules.txt"
            # Directly overwrite Confusing Intent Determination Rules table
            with open(rules_file, "w", encoding="utf-8") as f:
                f.write(self.similar_intent_rules)
            print(f"    ✨ Rules table cleaned and overwritten (Cleaner): {rules_file}")

        except Exception as e:
            print(f"❌ Failed to save Cleaner tables to file: {e}")

    def train(self, max_samples: Optional[int] = None, save_interval: int = 50, resume_from: int = 0):
        """
        Execute training loop

        Args:
            max_samples: Maximum number of training samples (None means all)
            save_interval: Save checkpoint every N batches (when batch_size>1) or every N samples (when batch_size=1)
            resume_from: Which sample to start from (0 means start from beginning)
        """
        samples_to_process = self.train_data[resume_from:max_samples] if max_samples else self.train_data[resume_from:]

        # If resuming training, reload existing table content
        if resume_from > 0:
            print(f"\n{'='*60}")
            print(f"📂 Resuming training from sample {resume_from + 1}...")
            print(f"   Reloading existing rules table...")
            print(f"{'='*60}\n")
            self.similar_intent_rules = self._load_similar_intent_rules()

        # Initialize batch-related variables
        self.batch_sample_count = 0

        # Debug: verify batch_size
        print(f"\n{'='*60}")
        print(f"[TRAIN DEBUG] batch_size before training starts: {self.batch_size}")
        print(f"{'='*60}\n")

        for idx, item in enumerate(samples_to_process):
            # Calculate global sample index (considering resume)
            global_idx = resume_from + idx
            print(f"\n{'='*60}")
            print(f"Processing progress: {global_idx + 1}/{len(self.train_data)}")
            print(f"{'='*60}")

            # Get true labels (support new data format)
            # New format: multimodal_intent_label (English), multimodal_sentiment_label (numeric)
            # Old format: intent (numeric), sentiment (numeric)

            # Prefer new format (English labels)
            if 'multimodal_intent_label' in item:
                true_intent = item.get('multimodal_intent_label', '')  # English label
            else:
                # Compatible with old format (numeric ID)
                true_intent_id = item.get('intent', -1)
                true_intent = INTENT_ID_TO_EN.get(true_intent_id, '')

            # Get sentiment label
            if 'multimodal_sentiment_label' in item:
                true_sentiment = item.get('multimodal_sentiment_label', -1)
            else:
                true_sentiment = item.get('sentiment', -1)

            # Validate label validity
            if not true_intent or true_sentiment == -1:
                print(f"⚠ Missing true labels (intent={true_intent}, sentiment={true_sentiment}), skipping this sample")
                continue

            # Call learner
            print("📚 Calling Learner...")
            model_response, pred_intent, pred_sentiment = self.call_learner(item)

            if not model_response:
                print("❌ Learner returned empty response")
                continue

            # Save Learner response to json (including complete sample information)
            self._save_agent_response_json(
                idx + 1, "learner", model_response,
                item=item,
                pred_intent=pred_intent,
                pred_sentiment=pred_sentiment,
                true_intent=true_intent,
                true_sentiment=true_sentiment
            )

            # Check if prediction is correct
            is_correct = (pred_intent == true_intent)

            if is_correct:
                print("✅ Prediction correct!")
                self.training_stats["correct_count"] += 1
            else:
                print("❌ Prediction incorrect!")
                print(f"   Predicted: intent={pred_intent}, sentiment={pred_sentiment}")
                print(f"   True: intent={true_intent}, sentiment={true_sentiment}")
                self.training_stats["error_count"] += 1

                # Add failed sample to current batch's error list
                self.batch_errors.append({
                    'item': item,
                    'model_response': model_response,
                    'pred_intent': pred_intent,
                    'pred_sentiment': pred_sentiment,
                    'true_intent': true_intent,
                    'true_sentiment': true_sentiment
                })

            # Increment batch count
            self.batch_sample_count += 1

            # Debug: print batch status
            print(f"[DEBUG] batch_size={self.batch_size}, batch_sample_count={self.batch_sample_count}, is_correct={is_correct}")
            print(f"[DEBUG] Condition check -> batch_size > 1? {self.batch_size > 1}  |  batch_sample_count >= batch_size? {self.batch_sample_count >= self.batch_size}")

            # When batch is full, process error samples and flush to main table
            if self.batch_size > 1 and self.batch_sample_count >= self.batch_size:
                print(f"[DEBUG] ✓ Entering batch processing branch")
                print(f"\n{'='*60}")
                print(f"🔄 Batch is full ({self.batch_sample_count} samples), preparing to process...")
                print(f"{'='*60}")

                # If there are failed samples in this batch, call optimizer and global regularizer
                if self.batch_errors:
                    print(f"📊 Found {len(self.batch_errors)} failed samples, calling Optimizer...")
                    optimizer_response = self.call_optimizer(self.batch_errors)

                    # Save Optimizer response to json
                    if optimizer_response:
                        batch_json_file = f"agent_responses/batch_{self.batch_count + 1}_optimizer.json"
                        try:
                            os.makedirs("agent_responses", exist_ok=True)
                            response_data = {
                                "batch": self.batch_count + 1,
                                "error_samples_count": len(self.batch_errors),
                                "type": "optimizer",
                                "optimizer_response": optimizer_response,
                                "timestamp": time.time()
                            }
                            with open(batch_json_file, 'a', encoding='utf-8') as f:
                                json.dump(response_data, f, ensure_ascii=False, indent=2)
                            print(f"✓ Optimizer response saved: {batch_json_file}")
                        except Exception as e:
                            print(f"⚠️  Failed to save Optimizer response: {e}")

                    # Extract table content from Optimizer response, update memory variables and ablation files
                    if optimizer_response:
                        print("📊 Extracting table content from Optimizer response...")
                        self._update_tables_from_response(optimizer_response, "optimizer")

                    # Immediately call global Regularizer after each Optimizer
                    if optimizer_response:
                        print("🌍 Immediately calling global Regularizer for optimization after each Optimizer...")
                        global_regularizer_response = self.call_regularizer(is_global=True)

                        # Process global Regularizer response
                        if global_regularizer_response:
                            print("📊 Extracting table content from global Regularizer response...")
                            self._update_tables_from_response(global_regularizer_response, "regularizer_global")

                            # Save global Regularizer response to json
                            batch_json_file = f"agent_responses/batch_{self.batch_count + 1}_global_regularizer.json"
                            try:
                                os.makedirs("agent_responses", exist_ok=True)
                                response_data = {
                                    "batch": self.batch_count + 1,
                                    "error_samples_count": len(self.batch_errors),
                                    "type": "global_regularizer",
                                    "regularizer_response": global_regularizer_response,
                                    "timestamp": time.time()
                                }
                                with open(batch_json_file, 'a', encoding='utf-8') as f:
                                    json.dump(response_data, f, ensure_ascii=False, indent=2)
                                print(f"✓ Global Regularizer response saved: {batch_json_file}")
                            except Exception as e:
                                print(f"⚠️  Failed to save global Regularizer response: {e}")

                            # Call Cleaner for further cleaning after Regularizer
                            cleaner_response = self.call_cleaner()

                            # Process Cleaner response
                            if cleaner_response:
                                print("📊 Extracting table content from Cleaner response...")
                                self._update_tables_from_response(cleaner_response, "cleaner")

                                # Save Cleaner response to json
                                batch_json_file = f"agent_responses/batch_{self.batch_count + 1}_cleaner.json"
                                try:
                                    os.makedirs("agent_responses", exist_ok=True)
                                    response_data = {
                                        "batch": self.batch_count + 1,
                                        "error_samples_count": len(self.batch_errors),
                                        "type": "cleaner",
                                        "cleaner_response": cleaner_response,
                                        "timestamp": time.time()
                                    }
                                    with open(batch_json_file, 'a', encoding='utf-8') as f:
                                        json.dump(response_data, f, ensure_ascii=False, indent=2)
                                    print(f"✓ Cleaner response saved: {batch_json_file}")
                                except Exception as e:
                                    print(f"⚠️  Failed to save Cleaner response: {e}")

                    # Clear error list
                    self.batch_errors = []
                else:
                    print(f"✓ All samples in this batch predicted correctly, skipping Optimizer and Regularizer")

                self.batch_sample_count = 0

            # Original logic batch_size == 1
            elif self.batch_size == 1 and not is_correct:
                print(f"[DEBUG] ✓ Entering per-sample processing branch")
                print(f"🔧 Calling Optimizer...")
                optimizer_response = self.call_optimizer(
                    [{
                        'item': item,
                        'model_response': model_response,
                        'pred_intent': pred_intent,
                        'pred_sentiment': pred_sentiment,
                        'true_intent': true_intent,
                        'true_sentiment': true_sentiment
                    }]
                )

                # Save Optimizer response to json (including complete sample information)
                if optimizer_response:
                    self._save_agent_response_json(
                        idx + 1, "optimizer", optimizer_response,
                        item=item,
                        pred_intent=pred_intent,
                        pred_sentiment=pred_sentiment,
                        true_intent=true_intent,
                        true_sentiment=true_sentiment
                    )

                if optimizer_response:
                    # Extract table content from Optimizer response, update memory variables and ablation files
                    print("📊 Extracting table content from Optimizer response...")
                    self._update_tables_from_response(optimizer_response, "optimizer")

                    # Immediately call global Regularizer after each Optimizer
                    print("🌍 Immediately calling global Regularizer for optimization after each Optimizer...")
                    global_regularizer_response = self.call_regularizer(is_global=True)

                    # Save global Regularizer response to json (including complete sample information)
                    if global_regularizer_response:
                        self._save_agent_response_json(
                            idx + 1, "global_regularizer", global_regularizer_response,
                            item=item,
                            pred_intent=pred_intent,
                            pred_sentiment=pred_sentiment,
                            true_intent=true_intent,
                            true_sentiment=true_sentiment
                        )

                        # Extract table content from global Regularizer response
                        print("📊 Extracting table content from global Regularizer response...")
                        self._update_tables_from_response(global_regularizer_response, "regularizer_global")

                        # Call Cleaner for further cleaning after Regularizer
                        cleaner_response = self.call_cleaner()

                        # Save Cleaner response to json (including complete sample information)
                        if cleaner_response:
                            self._save_agent_response_json(
                                idx + 1, "cleaner", cleaner_response,
                                item=item,
                                pred_intent=pred_intent,
                                pred_sentiment=pred_sentiment,
                                true_intent=true_intent,
                                true_sentiment=true_sentiment
                            )

                            # Extract table content from Cleaner response
                            print("📊 Extracting table content from Cleaner response...")
                            self._update_tables_from_response(cleaner_response, "cleaner")

            # Periodically save checkpoints
            # Regardless of batch_size, save every save_interval samples
            # Note: After each Optimizer, global Regularizer has already been called, here only save checkpoint files
            should_checkpoint = (global_idx + 1) % save_interval == 0

            if should_checkpoint:
                print(f"\n{'='*60}")
                print(f"💾 Saving checkpoint at sample {global_idx + 1}...")
                print(f"{'='*60}")

                # Save checkpoint (based on tables already optimized by global Regularizer)
                self._save_feature_and_rules(global_idx + 1)
                print(f"✅ Checkpoint for step {global_idx + 1} completed")

            # Add delay to avoid API rate limiting
            time.sleep(1)

        # If there are remaining unflushed batches, process and flush
        if self.batch_size > 1 and self.batch_sample_count > 0:
            print(f"\n{'='*60}")
            print(f"🔄 Training ended, processing remaining {self.batch_sample_count} samples...")
            print(f"{'='*60}")

            # Process error samples in remaining batch
            if self.batch_errors:
                print(f"📊 Found {len(self.batch_errors)} failed samples, calling Optimizer...")
                optimizer_response = self.call_optimizer(self.batch_errors)

                if optimizer_response:
                    # Extract table content from Optimizer response
                    print("📊 Extracting table content from Optimizer response...")
                    self._update_tables_from_response(optimizer_response, "optimizer")

                    # Immediately call global Regularizer after each Optimizer
                    print("🌍 Immediately calling global Regularizer for optimization after each Optimizer...")
                    global_regularizer_response = self.call_regularizer(is_global=True)

                    # Extract table content from global Regularizer response
                    if global_regularizer_response:
                        print("📊 Extracting table content from global Regularizer response...")
                        self._update_tables_from_response(global_regularizer_response, "regularizer_global")

                        # Call Cleaner for further cleaning after Regularizer
                        cleaner_response = self.call_cleaner()

                        # Extract table content from Cleaner response
                        if cleaner_response:
                            print("📊 Extracting table content from Cleaner response...")
                            self._update_tables_from_response(cleaner_response, "cleaner")

                self.batch_errors = []
            else:
                print(f"✓ All samples in remaining batch predicted correctly, skipping Optimizer and Regularizer")

        # Final save
        print(f"\n💾 Saving final tables...")
        self._save_feature_and_rules("final")

        # Print training statistics
        self._print_training_stats()

    def _print_training_stats(self):
        """Print training statistics"""
        print(f"\n{'='*60}")
        print("🎓 Training completed! Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {self.training_stats['total_samples']}")
        print(f"Correct predictions: {self.training_stats['correct_count']}")
        print(f"Incorrect predictions: {self.training_stats['error_count']}")
        print(f"Optimizer calls: {self.training_stats['optimizer_calls']}")
        print(f"Regularizer calls: {self.training_stats['regularizer_calls']}")
        print(f"Cleaner calls: {self.training_stats['cleaner_calls']}")

        if self.training_stats['total_samples'] > 0:
            accuracy = self.training_stats['correct_count'] / self.training_stats['total_samples']
            print(f"Accuracy: {accuracy:.2%}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Configuration parameters
    TRAIN_JSON_PATH = "train.json"
    BASE_IMAGE_PATH = "/root/autodl-fs/stickers"
    BATCH_SIZE = 2  # ← Adjust batch_size here (1 means per-sample, >1 means batch processing)

    # Debug: confirm main program variables
    print(f"\n{'='*60}")
    print(f"[MAIN DEBUG] BATCH_SIZE set to: {BATCH_SIZE}")
    print(f"[MAIN DEBUG] Preparing to pass to LearningTrainingSystem: batch_size={BATCH_SIZE}")
    print(f"{'='*60}\n")

    # Initialize training system
    print("🚀 Initializing learning training system...")
    system = LearningTrainingSystem(TRAIN_JSON_PATH, BASE_IMAGE_PATH, batch_size=BATCH_SIZE)

    # Start training
    print("\n🎓 Starting training...")
    system.train(save_interval=20, resume_from = 0)

    print("\n✅ Learning training system execution completed!")
