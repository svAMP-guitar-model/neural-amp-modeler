import os
import json
import random
import pandas as pd

# for now only using poly, but in real test, we'd do a split btwn the two csvs
# vars for file paths
wav_data_repo_path = "/Users/vishnuvelayuthan/computer-science/school-comp-sci/cs444/project/svAMP-data-prep/IDMT-SMT-AUDIO-EFFECTS/IDMT-SMT-AUDIO-EFFECTS/Gitarre polyphon/Samples"
wav_csv_path = "/Users/vishnuvelayuthan/computer-science/school-comp-sci/cs444/project/svAMP-data-prep/audio_effects_poly_mapping.csv"

output_dataset_file_path = "data/multi_prompt_dataset.json"
num_examples = 300 

# vars for model and ds gen
embedding_size = 768
ny = 1024
seed = 1922 # Set to an int to get deterministic shuffling

splits = {
    "train": 90,
    "validation": 10
}


def _resolve_path(p: str) -> str:
    return os.path.join(wav_data_repo_path, p) + ".wav"


def _make_entries(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
        x_path = _resolve_path(str(row["clean_path"]))
        y_path = _resolve_path(str(row["processed_path"]))
        prompt =  str(row["prompt"]).strip()
        records.append({
            "x_path": x_path,
            "y_path": y_path,
            "prompt": prompt
        })
    return records


def _split_entries(entries: list, split_cfg: dict) -> tuple:
    if not entries:
        return [], []

    # Deterministic split without randomness (preserve CSV order)
    n = len(entries)
    n_train = int(round(n * (split_cfg.get("train", 0) / 100.0)))
    n_train = min(max(n_train, 0), n)
    train = entries[:n_train]
    val = entries[n_train:]
    return train, val


def generate_dataset_json():
    df = pd.read_csv(wav_csv_path)
    entries = _make_entries(df)

    rng = random.Random(int(seed))
    entries = list(entries)
    rng.shuffle(entries)

    entries = entries[:int(num_examples)]
    train, validation = _split_entries(entries, splits)

    dataset_json = {
        "_notes": [
            "Auto-generated from CSV using nam_full_configs/dataset_generator.py"
        ],
        "type": "prompt_dataset",
        "common": {
            "start_seconds": None,
            "stop_seconds": None,
            "embedding_size": embedding_size,
            "delay": 0,
            "ny": ny
        },
        "train": train,
        "validation": validation
    }

    os.makedirs(os.path.dirname(output_dataset_file_path) or ".", exist_ok=True)
    with open(output_dataset_file_path, "w") as f:
        json.dump(dataset_json, f, indent=2)


if __name__ == "__main__":
    generate_dataset_json()

