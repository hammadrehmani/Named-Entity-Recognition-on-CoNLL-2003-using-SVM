"""Data loading module for CoNLL-2003 Named Entity Recognition dataset.

Supports multiple data sources and formats:
1. Kaggle datasets with .txt files (/kaggle/input/**/train.txt, valid.txt, test.txt)
2. Kaggle datasets with .parquet files (/kaggle/input/**/train.parquet, ...)
3. Local files (./data/train.parquet, train.txt, ...)
4. Direct HuggingFace Hub fallback (lhoestq/conll2003)
"""

import glob
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Canonical CoNLL-2003 BIO NER tags
NER_TAG_NAMES = [
    'O',       # 0: Outside of any entity
    'B-PER',   # 1: Beginning of Person name
    'I-PER',   # 2: Continuation of Person name
    'B-ORG',   # 3: Beginning of Organization name
    'I-ORG',   # 4: Continuation of Organization name
    'B-LOC',   # 5: Beginning of Location name
    'I-LOC',   # 6: Continuation of Location name
    'B-MISC',  # 7: Beginning of Miscellaneous entity
    'I-MISC'   # 8: Continuation of Miscellaneous entity
]

# Canonical Penn Treebank Part-of-Speech tags in CoNLL-2003
POS_TAG_NAMES = [
    '"', "''", '#', '$', '(', ')', ',', '.', ':', '``', 'CC', 'CD', 'DT', 'EX', 'FW',
    'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'NN|SYM',
    'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'
]

NER_ID2LABEL = {i: tag for i, tag in enumerate(NER_TAG_NAMES)}
NER_LABEL2ID = {tag: i for i, tag in enumerate(NER_TAG_NAMES)}
POS_LABEL2ID = {tag: i for i, tag in enumerate(POS_TAG_NAMES)}


def parse_conll_txt(filepath: str) -> pd.DataFrame:
    """Parse a classic CoNLL-2003 formatted text file (Token POS Chunk NER).
    
    Args:
        filepath: Path to the .txt file.
        
    Returns:
        pd.DataFrame with 'id', 'tokens', 'pos_tags', and 'ner_tags' columns.
    """
    sentences = []
    tokens: List[str] = []
    pos_tags: List[int] = []
    ner_tags: List[int] = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({
                        'tokens': tokens,
                        'pos_tags': pos_tags,
                        'ner_tags': ner_tags
                    })
                    tokens, pos_tags, ner_tags = [], [], []
                continue

            parts = line.split()
            if len(parts) < 2 or parts[0] == '-DOCSTART-':
                continue

            tok = parts[0]
            pos = parts[1] if len(parts) >= 2 else 'NN'
            ner = parts[-1] if len(parts) >= 4 else (parts[2] if len(parts) >= 3 else 'O')

            tokens.append(tok)
            pos_tags.append(POS_LABEL2ID.get(pos, 21))  # 21 = NN default
            ner_tags.append(NER_LABEL2ID.get(ner, 0))   # 0 = O default

        if tokens:
            sentences.append({
                'tokens': tokens,
                'pos_tags': pos_tags,
                'ner_tags': ner_tags
            })

    df = pd.DataFrame(sentences)
    df['id'] = [str(i) for i in range(len(df))]
    return df


def find_conll_dataset() -> Tuple[Optional[str], Optional[str]]:
    """Scan Kaggle input directories and local paths for CoNLL-2003 files.
    
    Returns:
        Tuple of (format_type, directory_path) where format_type is 'txt' or 'parquet'.
    """
    search_roots = [
        "/kaggle/input",
        "/kaggle/input/conll003-english-version",
        "/kaggle/input/conll003",
        "/kaggle/input/conll2003",
        "../input",
        "./data",
        "data",
        "."
    ]

    for root in search_roots:
        if not os.path.exists(root):
            continue

        # 1. Search for .txt dataset files (Kaggle standard CoNLL text format)
        txt_matches = glob.glob(os.path.join(root, "**/train.txt"), recursive=True)
        if not txt_matches and os.path.exists(os.path.join(root, "train.txt")):
            txt_matches = [os.path.join(root, "train.txt")]

        for train_file in txt_matches:
            folder = os.path.dirname(train_file)
            test_candidates = [os.path.join(folder, "test.txt"), os.path.join(folder, "testa.txt")]
            if any(os.path.exists(t) for t in test_candidates):
                return 'txt', folder

        # 2. Search for .parquet dataset files
        parquet_matches = glob.glob(os.path.join(root, "**/train.parquet"), recursive=True)
        if not parquet_matches and os.path.exists(os.path.join(root, "train.parquet")):
            parquet_matches = [os.path.join(root, "train.parquet")]

        for train_file in parquet_matches:
            folder = os.path.dirname(train_file)
            if os.path.exists(os.path.join(folder, "test.parquet")):
                return 'parquet', folder

    return None, None


def load_conll2003(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits into pandas DataFrames.
    
    Automatically discovers Kaggle input directories (/kaggle/input/**),
    local folders, and handles both .txt and .parquet formats.
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    fmt = None
    folder = None

    if data_dir and os.path.exists(data_dir):
        if os.path.exists(os.path.join(data_dir, "train.txt")):
            fmt, folder = 'txt', data_dir
        elif os.path.exists(os.path.join(data_dir, "train.parquet")):
            fmt, folder = 'parquet', data_dir

    if not fmt:
        fmt, folder = find_conll_dataset()

    if fmt == 'txt' and folder:
        print(f"[DataLoader] Detected CoNLL-2003 text dataset in: {folder}")
        train_path = os.path.join(folder, "train.txt")
        test_path = os.path.join(folder, "test.txt")
        
        # Check validation candidates (valid.txt, val.txt, dev.txt, testb.txt)
        val_candidates = ["valid.txt", "val.txt", "dev.txt", "testa.txt"]
        val_path = None
        for v in val_candidates:
            cand = os.path.join(folder, v)
            if os.path.exists(cand):
                val_path = cand
                break

        train_df = parse_conll_txt(train_path)
        val_df = parse_conll_txt(val_path) if val_path else pd.DataFrame()
        test_df = parse_conll_txt(test_path)

    elif fmt == 'parquet' and folder:
        print(f"[DataLoader] Detected CoNLL-2003 parquet dataset in: {folder}")
        train_df = pd.read_parquet(os.path.join(folder, "train.parquet"))
        val_p = os.path.join(folder, "validation.parquet")
        val_df = pd.read_parquet(val_p) if os.path.exists(val_p) else pd.DataFrame()
        test_df = pd.read_parquet(os.path.join(folder, "test.parquet"))

    else:
        print("[DataLoader] Local files not found. Fetching canonical dataset from Hugging Face Hub...")
        try:
            from datasets import load_dataset
            ds = load_dataset("lhoestq/conll2003")
            train_df = ds['train'].to_pandas()
            val_df = ds['validation'].to_pandas()
            test_df = ds['test'].to_pandas()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CoNLL-2003 dataset. Please verify Kaggle dataset is added or files exist. Error: {e}"
            )

    # Normalize list formats
    for df in [train_df, val_df, test_df]:
        if not df.empty:
            df['tokens'] = df['tokens'].apply(list)
            df['pos_tags'] = df['pos_tags'].apply(list)
            df['ner_tags'] = df['ner_tags'].apply(list)

    print(f"[DataLoader] Successfully loaded splits: train={len(train_df):,} sentences, "
          f"val={len(val_df):,} sentences, test={len(test_df):,} sentences.")
    return train_df, val_df, test_df
