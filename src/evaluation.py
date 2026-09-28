"""Evaluation and Error Analysis Module for Named Entity Recognition.

Computes both Token-Level and Entity-Level (Span-Based) metrics using
scikit-learn and seqeval, with comprehensive error categorization and inspection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
try:
    from seqeval.metrics import (
        classification_report as seqeval_report,
        f1_score as seqeval_f1,
        precision_score as seqeval_precision,
        recall_score as seqeval_recall
    )
except (ImportError, ModuleNotFoundError):
    def _extract_bio_entities(seq: List[str]) -> set:
        entities = []
        curr = None
        for i, tag in enumerate(seq):
            if tag.startswith('B-'):
                if curr:
                    entities.append(curr)
                curr = (tag[2:], i, i)
            elif tag.startswith('I-'):
                etype = tag[2:]
                if curr and curr[0] == etype:
                    curr = (curr[0], curr[1], i)
                else:
                    if curr:
                        entities.append(curr)
                    curr = (etype, i, i)
            else:
                if curr:
                    entities.append(curr)
                curr = None
        if curr:
            entities.append(curr)
        return set(entities)

    def seqeval_f1(y_true: List[List[str]], y_pred: List[List[str]], average: str = 'micro') -> float:
        tp, fp, fn = 0, 0, 0
        for yt, yp in zip(y_true, y_pred):
            te, pe = _extract_bio_entities(yt), _extract_bio_entities(yp)
            tp += len(te & pe)
            fp += len(pe - te)
            fn += len(te - pe)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def seqeval_precision(y_true: List[List[str]], y_pred: List[List[str]], average: str = 'micro') -> float:
        tp, fp = 0, 0
        for yt, yp in zip(y_true, y_pred):
            te, pe = _extract_bio_entities(yt), _extract_bio_entities(yp)
            tp += len(te & pe)
            fp += len(pe - te)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def seqeval_recall(y_true: List[List[str]], y_pred: List[List[str]], average: str = 'micro') -> float:
        tp, fn = 0, 0
        for yt, yp in zip(y_true, y_pred):
            te, pe = _extract_bio_entities(yt), _extract_bio_entities(yp)
            tp += len(te & pe)
            fn += len(te - pe)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def seqeval_report(y_true: List[List[str]], y_pred: List[List[str]], digits: int = 4) -> str:
        counts: Dict[str, Dict[str, int]] = {}
        for yt, yp in zip(y_true, y_pred):
            te, pe = _extract_bio_entities(yt), _extract_bio_entities(yp)
            for etype, s, e in te:
                counts.setdefault(etype, {'tp': 0, 'fp': 0, 'fn': 0})
                if (etype, s, e) in pe:
                    counts[etype]['tp'] += 1
                else:
                    counts[etype]['fn'] += 1
            for etype, s, e in pe:
                counts.setdefault(etype, {'tp': 0, 'fp': 0, 'fn': 0})
                if (etype, s, e) not in te:
                    counts[etype]['fp'] += 1
        lines = [f"{'':>15}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}"]
        total_tp, total_fp, total_fn = 0, 0, 0
        for etype in sorted(counts.keys()):
            c = counts[etype]
            tp, fp, fn = c['tp'], c['fp'], c['fn']
            supp = tp + fn
            total_tp += tp
            total_fp += fp
            total_fn += fn
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            lines.append(f"{etype:>15}  {p:>9.4f}  {r:>9.4f}  {f:>9.4f}  {supp:>9d}")
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        lines.append(f"\n{'micro avg':>15}  {micro_p:>9.4f}  {micro_r:>9.4f}  {micro_f:>9.4f}  {total_tp + total_fn:>9d}")
        return "\n".join(lines)

from .data_loader import NER_TAG_NAMES


def compute_token_level_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate token-level classification metrics."""
    if labels is None:
        labels = [tag for tag in NER_TAG_NAMES if tag in set(y_true) or tag in set(y_pred)]
        
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        'accuracy': acc,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'macro_f1': f1_macro,
        'weighted_precision': prec_weighted,
        'weighted_recall': rec_weighted,
        'weighted_f1': f1_weighted,
        'classification_report_dict': report_dict,
        'classification_report_text': report_text,
        'confusion_matrix': cm,
        'labels': labels
    }


def reconstruct_sequences(
    y_flat: List[str],
    sentence_lengths: List[int]
) -> List[List[str]]:
    """Reconstruct sentence-level tag sequences from a flattened list of token labels."""
    sequences: List[List[str]] = []
    idx = 0
    y_list = list(y_flat)
    for length in sentence_lengths:
        sequences.append(y_list[idx : idx + length])
        idx += length
    return sequences


def compute_entity_level_metrics(
    y_true_flat: List[str],
    y_pred_flat: List[str],
    sentence_lengths: List[int]
) -> Dict[str, Any]:
    """Calculate strict span-based entity-level metrics via seqeval.
    
    In Named Entity Recognition, entity-level evaluation requires exact matching of
    both entity boundaries (e.g. multi-word names) and category types.
    """
    y_true_seq = reconstruct_sequences(y_true_flat, sentence_lengths)
    y_pred_seq = reconstruct_sequences(y_pred_flat, sentence_lengths)
    
    p = seqeval_precision(y_true_seq, y_pred_seq)
    r = seqeval_recall(y_true_seq, y_pred_seq)
    f1 = seqeval_f1(y_true_seq, y_pred_seq)
    rep_text = seqeval_report(y_true_seq, y_pred_seq, digits=4)
    
    return {
        'entity_precision': p,
        'entity_recall': r,
        'entity_f1': f1,
        'entity_report_text': rep_text,
        'true_sequences': y_true_seq,
        'pred_sequences': y_pred_seq
    }


def analyze_errors(
    df: pd.DataFrame,
    y_true_flat: List[str],
    y_pred_flat: List[str],
    sentence_lengths: List[int],
    max_examples: int = 15
) -> Dict[str, Any]:
    """Perform detailed qualitative and quantitative error analysis."""
    y_true_seq = reconstruct_sequences(y_true_flat, sentence_lengths)
    y_pred_seq = reconstruct_sequences(y_pred_flat, sentence_lengths)
    
    error_sentences = []
    confusion_counts = {
        'false_positives (O -> Entity)': 0,
        'false_negatives (Entity -> O)': 0,
        'entity_type_confusion': 0,
        'boundary_confusion': 0
    }
    
    for s_idx, (tokens, t_seq, p_seq) in enumerate(zip(df['tokens'], y_true_seq, y_pred_seq)):
        has_error = False
        token_comparisons = []
        for i, (tok, t_tag, p_tag) in enumerate(zip(tokens, t_seq, p_seq)):
            match = (t_tag == p_tag)
            if not match:
                has_error = True
                if t_tag == 'O' and p_tag != 'O':
                    confusion_counts['false_positives (O -> Entity)'] += 1
                elif t_tag != 'O' and p_tag == 'O':
                    confusion_counts['false_negatives (Entity -> O)'] += 1
                elif t_tag[2:] == p_tag[2:] and t_tag[:2] != p_tag[:2]:
                    confusion_counts['boundary_confusion'] += 1
                else:
                    confusion_counts['entity_type_confusion'] += 1
                    
            token_comparisons.append({
                'token': tok,
                'true_tag': t_tag,
                'pred_tag': p_tag,
                'is_correct': match
            })
            
        if has_error and len(error_sentences) < max_examples:
            error_sentences.append({
                'sentence_index': s_idx,
                'tokens': tokens,
                'token_comparisons': token_comparisons
            })
            
    return {
        'error_counts_by_category': confusion_counts,
        'sampled_error_sentences': error_sentences
    }
