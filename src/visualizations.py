"""Visualization Module for Named Entity Recognition EDA and Model Evaluation.

Generates publication-quality figures with professional styling, clear typography,
and informative annotations.
"""

import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set overall design aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def plot_dataset_splits_overview(
    splits_dict: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot sentence and token counts across train, validation, and test splits."""
    split_names = list(splits_dict.keys())
    sentence_counts = [len(df) for df in splits_dict.values()]
    token_counts = [sum(len(toks) for toks in df['tokens']) for df in splits_dict.values()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    colors = ['#2b5c8f', '#4682b4', '#87ceeb']

    # Sentences bar chart
    bars1 = axes[0].bar(split_names, sentence_counts, color=colors, edgecolor='#1d3d60', width=0.55)
    axes[0].set_title("Number of Sentences per Split", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel("Sentence Count", fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + max(sentence_counts) * 0.015,
                     f"{yval:,}", ha='center', va='bottom', fontsize=10, fontweight='semibold')

    # Tokens bar chart
    bars2 = axes[1].bar(split_names, token_counts, color=['#2e8b57', '#3cb371', '#8fbc8f'], edgecolor='#1c5334', width=0.55)
    axes[1].set_title("Number of Tokens per Split", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_ylabel("Token Count", fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + max(token_counts) * 0.015,
                     f"{yval:,}", ha='center', va='bottom', fontsize=10, fontweight='semibold')

    plt.suptitle("CoNLL-2003 Dataset Overview", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_entity_distributions(
    train_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot distribution of entity labels excluding the background 'O' class."""
    from .data_loader import NER_TAG_NAMES
    
    tag_counts: Dict[str, int] = {tag: 0 for tag in NER_TAG_NAMES if tag != 'O'}
    for tags in train_df['ner_tags']:
        for t in tags:
            tag_name = NER_TAG_NAMES[t]
            if tag_name in tag_counts:
                tag_counts[tag_name] += 1

    df_counts = pd.DataFrame(list(tag_counts.items()), columns=['Tag', 'Count'])
    df_counts.sort_values(by='Count', ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    palette = sns.color_palette("mako", len(df_counts))
    bars = ax.bar(df_counts['Tag'], df_counts['Count'], color=palette, edgecolor='#333333', width=0.6)

    ax.set_title("Distribution of Named Entity Tags in Training Set (Excluding 'O')",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("NER Tag (BIO Format)", fontsize=11, labelpad=8)
    ax.set_ylabel("Token Frequency", fontsize=11, labelpad=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(df_counts['Count']) * 0.015,
                f"{height:,}", ha='center', va='bottom', fontsize=9, fontweight='semibold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_sentence_length_distribution(
    train_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot sentence length histogram with descriptive statistics."""
    lengths = [len(toks) for toks in train_df['tokens']]
    median_len = np.median(lengths)
    mean_len = np.mean(lengths)
    max_len = np.max(lengths)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.histplot(lengths, bins=40, kde=True, color='#2b5c8f', edgecolor='white', ax=ax)

    ax.axvline(median_len, color='#d9534f', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
    ax.axvline(mean_len, color='#f0ad4e', linestyle='-', linewidth=2, label=f'Mean: {mean_len:.1f}')

    ax.set_title("Sentence Length Distribution (Token Count per Sentence)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Tokens in Sentence", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xlim(0, min(max_len, 80))
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot normalized or raw confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, cbar=True, ax=ax,
                    linewidths=0.5, linecolor='#EEEEEE')
        ax.set_title("Normalized Confusion Matrix (Row Proportions)", fontsize=13, fontweight='bold', pad=12)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, cbar=True, ax=ax,
                    linewidths=0.5, linecolor='#EEEEEE')
        ax.set_title("Raw Confusion Matrix", fontsize=13, fontweight='bold', pad=12)

    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_per_class_metrics(
    report_dict: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot per-class Precision, Recall, and F1-score comparison."""
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    precisions = [report_dict[c]['precision'] for c in classes]
    recalls = [report_dict[c]['recall'] for c in classes]
    f1s = [report_dict[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.bar(x - width, precisions, width, label='Precision', color='#2b5c8f', edgecolor='#1d3d60')
    ax.bar(x, recalls, width, label='Recall', color='#3cb371', edgecolor='#1c5334')
    ax.bar(x + width, f1s, width, label='F1-Score', color='#f0ad4e', edgecolor='#ad721d')

    ax.set_title("Per-Class Performance on Test Set (Precision, Recall, F1)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Entity Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Score", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_error_type_distribution(
    error_counts: Dict[str, int],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot distribution of error categories."""
    categories = list(error_counts.keys())
    counts = list(error_counts.values())

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    palette = ['#d9534f', '#f0ad4e', '#5bc0de', '#9370db']
    bars = ax.barh(categories, counts, color=palette, edgecolor='#333333', height=0.55)

    ax.set_title("Breakdown of Error Categories on Test Set", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Misclassified Tokens", fontsize=11, labelpad=8)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(counts) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}", ha='left', va='center', fontsize=10, fontweight='semibold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    return fig
