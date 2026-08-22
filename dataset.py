import os
import sys

# Fix for Windows UnicodeEncodeError when printing Hindi (Devanagari) characters
if sys.platform.startswith('win') and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Path to the local pre-downloaded dataset file (relative to project root)
_LOCAL_DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_preview_english.txt")

def get_msmarco_passages(limit=100):
    """
    Loads passages from the local dataset_preview_english.txt file.
    Passages are separated by '---' lines. This avoids slow/throttled
    unauthenticated downloads from HuggingFace Hub.
    """
    print("Loading real dataset passages...")
    
    if not os.path.exists(_LOCAL_DATASET_FILE):
        print(f"WARNING: Local dataset file not found at {_LOCAL_DATASET_FILE}")
        return ["This is a fallback passage because the dataset file was not found."]
    
    with open(_LOCAL_DATASET_FILE, "r", encoding="utf-8") as f:
        raw = f.read()
    
    # Split on the '---' separator used in the file
    chunks = [p.strip() for p in raw.split("---") if p.strip()]
    
    passages = chunks[:limit]
    print(f"Loaded {len(passages)} passages from local file.")
    return passages


if __name__ == "__main__":
    passages = get_msmarco_passages(limit=5)
    print(f"Loaded {len(passages)} passages.")
    for p in passages:
        print(f"- {p[:100]}...")