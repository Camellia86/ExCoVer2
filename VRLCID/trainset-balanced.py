import json
import random
from collections import defaultdict

# ===================== 1. Core Configuration =====================
RANDOM_SEED = 2026
TARGET_TOTAL_NUM = 200
VAL_RATIO = 0.1
INPUT_FILE = "train.json"
OUTPUT_SERIAL = "train-uniform.json"
OUTPUT_VAL = "val.json"
OUTPUT_TRAIN_REST = "train-rest.json"

# ===================== 2. Intent Mapping Dictionary =====================
INTENT_ID_TO_EN = {v: k for k, v in {
    "Complain": 0, "Praise": 1, "Agree": 2, "Compromise": 3, "Query": 4,
    "Joke": 5, "Oppose": 6, "Inform": 7, "Ask for help": 8, "Greet": 9,
    "Taunt": 10, "Introduce": 11, "Guess": 12, "Leave": 13, "Advise": 14,
    "Flaunt": 15, "Criticize": 16, "Thank": 17, "Comfort": 18, "Apologize": 19
}.items()}
INTENT_EN_TO_ID = {v: k for k, v in INTENT_ID_TO_EN.items()}
VALID_INTENTS = list(INTENT_ID_TO_EN.values())
INTENT_COUNT = len(VALID_INTENTS)

# ===================== 3. Utility Functions =====================
def detect_file_format(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        return "array"
    except json.JSONDecodeError:
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
        return None

def load_full_data(file_path, file_format):
    full_data = []
    if file_format == "array":
        with open(file_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
    elif file_format == "lines":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        full_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return full_data

def split_train_val(full_data, val_ratio, seed):
    random.seed(seed)
    data_shuffled = random.sample(full_data, len(full_data))
    val_num = int(len(data_shuffled) * val_ratio)
    val_data = data_shuffled[:val_num]
    train_rest_data = data_shuffled[val_num:]
    return train_rest_data, val_data

def group_data_by_intent(data_list):
    intent_to_data = defaultdict(list)
    for data in data_list:
        intent_name = data.get("multimodal_intent_label", data.get("intent"))
        if intent_name in VALID_INTENTS:
            intent_to_data[intent_name].append(data)
    for intent in VALID_INTENTS:
        if not intent_to_data[intent]:
            raise ValueError(f"Error: Intent [{intent}] has no matching samples!")
    return intent_to_data

def save_data_by_format(data_list, file_path, file_format):
    if file_format == "array":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    elif file_format == "lines":
        with open(file_path, "w", encoding="utf-8") as f:
            for data in data_list:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

def calculate_intent_sample_num():
    intent_sample_num = {}
    base_num = TARGET_TOTAL_NUM // INTENT_COUNT
    remainder = TARGET_TOTAL_NUM % INTENT_COUNT

    for intent in VALID_INTENTS:
        intent_sample_num[intent] = base_num

    random.seed(RANDOM_SEED)
    remainder_intents = random.sample(VALID_INTENTS, remainder)
    for intent in remainder_intents:
        intent_sample_num[intent] += 1

    assert sum(intent_sample_num.values()) == TARGET_TOTAL_NUM
    return intent_sample_num

def sample_with_replacement(data_pool, sample_num, seed):
    random.seed(seed)
    sampled_data = []
    for _ in range(sample_num):
        sampled = random.choice(data_pool)
        sampled_data.append(sampled)
    return sampled_data

def generate_custom_serial_data(intent_to_data, intent_sample_num):
    random.seed(RANDOM_SEED)
    serial_data = []

    first_part = []
    for intent in VALID_INTENTS:
        one_sample = random.choice(intent_to_data[intent])
        first_part.append(one_sample)
    serial_data.extend(first_part)

    remaining_total = TARGET_TOTAL_NUM - len(first_part)
    if remaining_total < 0:
        raise ValueError(f"Target total {TARGET_TOTAL_NUM} is less than intent count {INTENT_COUNT}")

    intent_remaining_num = {intent: max(0, intent_sample_num[intent]-1) for intent in VALID_INTENTS}
    assert sum(intent_remaining_num.values()) == remaining_total

    second_part = []
    for intent in VALID_INTENTS:
        sample_num = intent_remaining_num[intent]
        if sample_num <= 0:
            continue
        sampled = sample_with_replacement(intent_to_data[intent], sample_num, RANDOM_SEED + hash(intent))
        second_part.extend(sampled)

    serial_data.extend(second_part)
    assert len(serial_data) == TARGET_TOTAL_NUM
    return serial_data

# ===================== 4. Main Function =====================
def main():
    file_format = detect_file_format(INPUT_FILE)
    if not file_format:
        print("Error: Invalid file format")
        return
    print(f"Detected file format: {file_format}")

    full_data = load_full_data(INPUT_FILE, file_format)
    print(f"Total original samples: {len(full_data)}")

    train_rest_data, val_data = split_train_val(full_data, VAL_RATIO, RANDOM_SEED)
    print(f"Validation set: {len(val_data)} | Remaining training set: {len(train_rest_data)}")

    save_data_by_format(val_data, OUTPUT_VAL, file_format)
    save_data_by_format(train_rest_data, OUTPUT_TRAIN_REST, file_format)
    print(f"Saved validation set: {OUTPUT_VAL}")
    print(f"Saved remaining training set: {OUTPUT_TRAIN_REST}")

    try:
        intent_to_data = group_data_by_intent(train_rest_data)
    except ValueError as e:
        print(f"Data grouping failed: {e}")
        return

    intent_sample_num = calculate_intent_sample_num()
    print("\nSample count per intent:")
    for intent in sorted(VALID_INTENTS):
        print(f"- {intent}: {intent_sample_num[intent]}")

    try:
        serial_data = generate_custom_serial_data(intent_to_data, intent_sample_num)
        print(f"\nGenerated uniform samples: {len(serial_data)}")
        first_n_intents = [d.get("multimodal_intent_label", d.get("intent")) for d in serial_data[:INTENT_COUNT]]
        if len(set(first_n_intents)) == INTENT_COUNT:
            print("Validation passed: Unique intents in first N samples")
        else:
            print("Validation failed: Duplicate intents in first N samples")
    except ValueError as e:
        print(f"Generation failed: {e}")
        return

    save_data_by_format(serial_data, OUTPUT_SERIAL, file_format)
    print(f"\nSaved uniform samples: {OUTPUT_SERIAL}")

    print("\n" + "="*50)
    print(f"Original: {len(full_data)} | Val: {len(val_data)} | Train rest: {len(train_rest_data)}")
    print(f"Final uniform samples: {len(serial_data)}")

if __name__ == "__main__":
    main()