import json
import random
from collections import defaultdict

# ===================== 1. Core Configuration =====================
# Fixed random seed for reproducibility
RANDOM_SEED = 2026
# Total number of samples to extract (core config: specify total count to extract)
TARGET_TOTAL_NUM = 100
# File path configuration
INPUT_FILE = "train.json"  # Original training data file
OUTPUT_SERIAL = "train-uniform.json"  # Custom serial result file

# ===================== 2. Intent Mapping Dictionary =====================
INTENT_ID_TO_EN = {v: k for k, v in {
    "Complain": 0, "Praise": 1, "Agree": 2, "Compromise": 3, "Query": 4,
    "Joke": 5, "Oppose": 6, "Inform": 7, "Ask for help": 8, "Greet": 9,
    "Taunt": 10, "Introduce": 11, "Guess": 12, "Leave": 13, "Advise": 14,
    "Flaunt": 15, "Criticize": 16, "Thank": 17, "Comfort": 18, "Apologize": 19
}.items()}
INTENT_EN_TO_ID = {v: k for k, v in INTENT_ID_TO_EN.items()}
# Valid intent list (all intents from mapping)
VALID_INTENTS = list(INTENT_ID_TO_EN.values())
INTENT_COUNT = len(VALID_INTENTS)  # Number of intent categories


# ===================== 3. Utility Functions (Reused + New) =====================
def detect_file_format(file_path):
    """Detect JSON file format: array (JSON array) / lines (JSON Lines) / None (invalid)"""
    try:
        # Try to read as JSON array
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        return "array"
    except json.JSONDecodeError:
        # Try to detect as JSON Lines
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        json.loads(line)
                return "lines"
        except:
            return None
    except Exception as e:
        print(f"Error detecting file format: {e}")
        return None


def load_data_by_format(file_path, file_format):
    """Load data by format, returns {intent_name: data_list}"""
    intent_to_data = defaultdict(list)
    if file_format == "array":
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            for data in data_list:
                intent_name = data.get("multimodal_intent_label", data.get("intent"))
                if intent_name in VALID_INTENTS:
                    intent_to_data[intent_name].append(data)
    elif file_format == "lines":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    intent_name = data.get("multimodal_intent_label", data.get("intent"))
                    if intent_name in VALID_INTENTS:
                        intent_to_data[intent_name].append(data)
                except json.JSONDecodeError:
                    continue
    # Validation: ensure all valid intents have data
    for intent in VALID_INTENTS:
        if not intent_to_data[intent]:
            raise ValueError(f"Error: Intent [{intent}] has no matching samples in the original data!")
    return intent_to_data


def save_data_by_format(data_list, file_path, file_format):
    """Save data according to original file format"""
    if file_format == "array":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    elif file_format == "lines":
        with open(file_path, "w", encoding="utf-8") as f:
            for data in data_list:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")


def calculate_intent_sample_num():
    """
    Calculate the number of samples to extract for each intent using uniform distribution:
    - Total evenly distributed, remainder randomly assigned
    Returns: {intent_name: sample_count}
    """
    intent_sample_num = {}

    # Uniform distribution: base_num = total_num // intent_count
    base_num = TARGET_TOTAL_NUM // INTENT_COUNT
    remainder = TARGET_TOTAL_NUM % INTENT_COUNT

    # Initialize all intents with base number
    for intent in VALID_INTENTS:
        intent_sample_num[intent] = base_num

    # Randomly distribute remainder (ensure total matches)
    random.seed(RANDOM_SEED)
    remainder_intents = random.sample(VALID_INTENTS, remainder)
    for intent in remainder_intents:
        intent_sample_num[intent] += 1

    # Validate total
    assert sum(intent_sample_num.values()) == TARGET_TOTAL_NUM, "Sum of sample counts per intent does not match target total!"
    return intent_sample_num


def sample_with_replacement(data_pool, sample_num, seed):
    """Sampling with replacement: extract specified number of samples from data pool (supports repeated sampling when data is insufficient)"""
    random.seed(seed)
    sampled_data = []
    for _ in range(sample_num):
        # Randomly select one with replacement
        sampled = random.choice(data_pool)
        sampled_data.append(sampled)
    return sampled_data


def generate_custom_serial_data(intent_to_data, intent_sample_num):
    """
    Generate custom serial data:
    1. First N samples: take 1 from each intent (ensure all intents are different)
    2. Remaining samples: sample with replacement by calculated count (minus the 1 already taken)
    """
    random.seed(RANDOM_SEED)
    serial_data = []

    # Step 1: Take 1 from each intent to form the first N samples (N = number of intents)
    first_part = []
    for intent in VALID_INTENTS:
        # Randomly select 1 (ensure each intent is taken only once)
        one_sample = random.choice(intent_to_data[intent])
        first_part.append(one_sample)
    serial_data.extend(first_part)

    # Step 2: Calculate remaining count to extract (total count - count from step 1)
    remaining_total = TARGET_TOTAL_NUM - len(first_part)
    if remaining_total < 0:
        raise ValueError(f"Target total {TARGET_TOTAL_NUM} is less than intent count {INTENT_COUNT}, cannot satisfy requirement of different intents in first N samples!")

    # Step 3: Calculate remaining count to extract for each intent (total - the 1 already taken)
    intent_remaining_num = {}
    for intent in VALID_INTENTS:
        intent_remaining_num[intent] = intent_sample_num[intent] - 1
        # Ensure count is non-negative
        intent_remaining_num[intent] = max(0, intent_remaining_num[intent])

    # Validate remaining total
    assert sum(intent_remaining_num.values()) == remaining_total, "Sum of remaining sample counts does not match!"

    # Step 4: Sample with replacement by remaining count and append to serial data
    second_part = []
    for intent in VALID_INTENTS:
        sample_num = intent_remaining_num[intent]
        if sample_num <= 0:
            continue
        # Sample with replacement
        sampled = sample_with_replacement(intent_to_data[intent], sample_num, RANDOM_SEED + hash(intent))
        second_part.extend(sampled)

    # Append to serial data
    serial_data.extend(second_part)

    # Final validation of total count
    assert len(serial_data) == TARGET_TOTAL_NUM, f"Generated data count {len(serial_data)} does not match target total {TARGET_TOTAL_NUM}!"
    return serial_data


# ===================== 4. Main Function =====================
def main():
    # 1. Detect original file format
    file_format = detect_file_format(INPUT_FILE)
    if not file_format:
        print("Error: Original file format is invalid, cannot be recognized as JSON array or JSON Lines!")
        return
    print(f"Detected original file format: {file_format}")

    # 2. Load data and group by intent
    try:
        intent_to_data = load_data_by_format(INPUT_FILE, file_format)
        print(f"Successfully loaded data, valid intent count: {len(intent_to_data)}")
    except ValueError as e:
        print(f"Failed to load data: {e}")
        return

    # 3. Calculate sample count for each intent
    intent_sample_num = calculate_intent_sample_num()
    print("\nSample count per intent:")
    for intent in sorted(VALID_INTENTS):
        print(f"- {intent}: {intent_sample_num[intent]}")

    # 4. Generate custom serial data
    try:
        serial_data = generate_custom_serial_data(intent_to_data, intent_sample_num)
        print(f"\nCustom serial data generation completed, total count: {len(serial_data)}")
        # Verify if first N samples have different intents
        first_n_intents = [d.get("multimodal_intent_label", d.get("intent")) for d in serial_data[:INTENT_COUNT]]
        if len(set(first_n_intents)) == INTENT_COUNT:
            print("✅ Validation passed: First N samples have all different intent labels")
        else:
            print("❌ Validation failed: First N samples have duplicate intent labels")
    except ValueError as e:
        print(f"Failed to generate data: {e}")
        return

    # 5. Save data
    save_data_by_format(serial_data, OUTPUT_SERIAL, file_format)
    print(f"\nFile saved successfully: {OUTPUT_SERIAL}")

    # 6. Print final statistics
    print("\n" + "=" * 50)
    print(f"Target extraction total: {TARGET_TOTAL_NUM}")
    print(f"Actual generated total: {len(serial_data)}")
    # Count actual number for each intent
    intent_actual_num = defaultdict(int)
    for d in serial_data:
        intent = d.get("multimodal_intent_label", d.get("intent"))
        intent_actual_num[intent] += 1
    print("Actual count per intent:")
    for intent in sorted(VALID_INTENTS):
        print(f"- {intent}: {intent_actual_num[intent]}")


if __name__ == "__main__":
    main()