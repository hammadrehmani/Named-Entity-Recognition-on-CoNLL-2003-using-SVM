"""Linguistic feature extraction for Named Entity Recognition (NER).

Extracts orthographic, morphological, syntactic, and contextual n-gram features
from sentence tokens to represent inputs in a high-dimensional sparse space.
"""

from typing import Any, Dict, List, Tuple
import string
import pandas as pd
from .data_loader import NER_TAG_NAMES, POS_TAG_NAMES

PUNCTUATION_SET = set(string.punctuation)


def extract_token_features(
    tokens: List[str],
    pos_tags: List[int],
    index: int
) -> Dict[str, Any]:
    """Extract a comprehensive dictionary of linguistic features for a single token.
    
    Args:
        tokens: List of string tokens in the sentence.
        pos_tags: List of integer POS tag IDs for the sentence.
        index: Position of the current token in the sentence.
        
    Returns:
        Dictionary of feature name -> feature value.
    """
    word = tokens[index]
    pos_id = pos_tags[index] if index < len(pos_tags) else -1
    pos = POS_TAG_NAMES[pos_id] if 0 <= pos_id < len(POS_TAG_NAMES) else "UNK"
    
    # Base token features
    features: Dict[str, Any] = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.length': len(word),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.has_digit': any(c.isdigit() for c in word),
        'word.has_punct': any(c in PUNCTUATION_SET for c in word),
        'word.has_hyphen': '-' in word,
        # Morphological prefixes and suffixes (sub-word cues)
        'word.prefix-1': word[:1],
        'word.prefix-2': word[:2] if len(word) >= 2 else word,
        'word.prefix-3': word[:3] if len(word) >= 3 else word,
        'word.suffix-1': word[-1:],
        'word.suffix-2': word[-2:] if len(word) >= 2 else word,
        'word.suffix-3': word[-3:] if len(word) >= 3 else word,
        # Syntactic Part-of-Speech information
        'pos': pos,
        'pos[:2]': pos[:2],
        # Positional flags
        'is_first': index == 0,
        'is_last': index == len(tokens) - 1,
    }
    
    # Context window: Preceding token (-1)
    if index > 0:
        prev_word = tokens[index - 1]
        prev_pos_id = pos_tags[index - 1] if index - 1 < len(pos_tags) else -1
        prev_pos = POS_TAG_NAMES[prev_pos_id] if 0 <= prev_pos_id < len(POS_TAG_NAMES) else "UNK"
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.istitle()': prev_word.istitle(),
            '-1:word.isupper()': prev_word.isupper(),
            '-1:word.has_punct': any(c in PUNCTUATION_SET for c in prev_word),
            '-1:pos': prev_pos,
            '-1:pos[:2]': prev_pos[:2],
        })
    else:
        features['BOS'] = True  # Beginning of sentence indicator
        
    # Context window: Subsequent token (+1)
    if index < len(tokens) - 1:
        next_word = tokens[index + 1]
        next_pos_id = pos_tags[index + 1] if index + 1 < len(pos_tags) else -1
        next_pos = POS_TAG_NAMES[next_pos_id] if 0 <= next_pos_id < len(POS_TAG_NAMES) else "UNK"
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.istitle()': next_word.istitle(),
            '+1:word.isupper()': next_word.isupper(),
            '+1:word.has_punct': any(c in PUNCTUATION_SET for c in next_word),
            '+1:pos': next_pos,
            '+1:pos[:2]': next_pos[:2],
        })
    else:
        features['EOS'] = True  # End of sentence indicator
        
    return features


def extract_features_from_dataframe(
    df: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
    """Extract token feature dictionaries, string labels, and sentence lengths from DataFrame.
    
    Args:
        df: Pandas DataFrame containing 'tokens', 'pos_tags', and 'ner_tags'.
        
    Returns:
        Tuple of (X_dicts, y_labels, sentence_lengths)
    """
    X_features: List[Dict[str, Any]] = []
    y_labels: List[str] = []
    sentence_lengths: List[int] = []
    
    for _, row in df.iterrows():
        tokens = row['tokens']
        pos = row['pos_tags']
        ner = row['ner_tags']
        sentence_lengths.append(len(tokens))
        
        for i in range(len(tokens)):
            feat = extract_token_features(tokens, pos, i)
            tag_id = ner[i]
            tag_name = NER_TAG_NAMES[tag_id] if 0 <= tag_id < len(NER_TAG_NAMES) else 'O'
            X_features.append(feat)
            y_labels.append(tag_name)
            
    return X_features, y_labels, sentence_lengths


def get_feature_rationales() -> Dict[str, str]:
    """Return educational rationale for each feature category in the NER pipeline."""
    return {
        "word.lower()": "Normalizes lexical identity across upper/lower cases to match words regardless of sentence position.",
        "word.istitle()": "Strongest single indicator for proper nouns (Names, Places, Organizations) in English text.",
        "word.isupper()": "Detects acronyms, corporate tickers, and geopolitical abbreviations (e.g., 'NATO', 'EU', 'BBC').",
        "word.isdigit() / has_digit": "Filters numeric artifacts, dates, sports results, and alphanumeric IDs.",
        "word.has_punct / has_hyphen": "Captures compound entities, hyphenated nationalities/locations ('Anglo-American', 'Al-Qaeda').",
        "word.prefix & suffix (1-3)": "Extracts morphological affixes (e.g. '-stan', '-land', '-ton' for LOC; '-ski', 'Mc-' for PER; '-corp', '-inc' for ORG).",
        "pos & pos[:2]": "Syntactic tag from Penn Treebank; NNP/NNPS identify proper nouns, JJ identifies nationality adjectives.",
        "context [-1, +1] tokens": "Captures surrounding grammatical patterns (e.g. honorifics 'Mr.', 'President' cue PER; prepositions 'in', 'at' cue LOC; verbs 'said', 'bought' cue ORG/PER).",
        "boundary flags (BOS, EOS, is_first)": "Disambiguates whether word-initial capitalization is merely sentence-initial or a genuine entity."
    }
