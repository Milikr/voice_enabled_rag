import os
import sys
import io
import warnings

# Fix for Windows UnicodeEncodeError when printing Hindi (Devanagari) characters
if sys.platform.startswith('win') and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress the symlink and authentication warnings
warnings.filterwarnings("ignore", module="huggingface_hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
import datasets

# Suppress the repo card metadata warning
datasets.utils.logging.set_verbosity_error()

def get_msmarco_passages(limit=10):
    """
    Loads the MS MARCO XI Hindi validation set and returns a list of translated passages.
    This parses the schema exactly as defined in ms_marco_translations.py.
    """
    print("Loading MS MARCO dataset (this might take a few moments)...")
    dataset = load_dataset(
        "ai4bharat/MSMARCO-XI", 
        data_files={"validation": "validation/hinval.parquet"}, 
        split="validation"
    )
    
    all_passages = []
    
    # Extract translated passages up to the limit
    for i, example in enumerate(dataset):
        if i >= limit:
            break
            
        # The schema in ms_marco_translations.py shows passages is a dict containing 'Translated_passages'
        passages_dict = example.get('passages', {})
        translated_passages = passages_dict.get('Translated_passages', [])
        
        for p in translated_passages:
            if p and p.strip():
                all_passages.append(p)
                
    return all_passages

if __name__ == "__main__":
    passages = get_msmarco_passages(limit=1)
    print(f"Loaded {len(passages)} passages.")
    for p in passages:
        print(f"- {p[:100]}...")