"""End-to-End Pipeline for CoNLL-2003 Named Entity Recognition using Support Vector Machine.

Author: Muhammad Hammad

Executes:
1. Dataset loading and verification
2. Exploratory Data Analysis (EDA) figure export
3. Token-level feature extraction
4. SVM model training with hyperparameter tuning
5. Token-level and Entity-level (Seqeval) evaluation
6. Detailed error analysis and figure generation
7. Serializing metrics and model artifact
"""

import json
import os
import time
import numpy as np
import pandas as pd

from src.data_loader import load_conll2003
from src.features import extract_features_from_dataframe, get_feature_rationales
from src.model import SVMNERModel
from src.evaluation import (
    compute_token_level_metrics,
    compute_entity_level_metrics,
    analyze_errors
)
from src.visualizations import (
    plot_dataset_splits_overview,
    plot_entity_distributions,
    plot_sentence_length_distribution,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_error_type_distribution
)


def main():
    print("=" * 80)
    print(" CoNLL-2003 Named Entity Recognition Pipeline (Linear SVM)")
    print("=" * 80)

    output_dir = "outputs"
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Dataset Loading
    print("\n[Step 1/6] Loading CoNLL-2003 Dataset...")
    train_df, val_df, test_df = load_conll2003()

    print(f"  Training set:   {len(train_df):,} sentences | {sum(len(t) for t in train_df['tokens']):,} tokens")
    print(f"  Validation set: {len(val_df):,} sentences | {sum(len(t) for t in val_df['tokens']):,} tokens")
    print(f"  Test set:       {len(test_df):,} sentences | {sum(len(t) for t in test_df['tokens']):,} tokens")

    # 2. EDA and Visualization Export
    print("\n[Step 2/6] Generating and Saving EDA Visualizations...")
    plot_dataset_splits_overview(
        {'Train': train_df, 'Validation': val_df, 'Test': test_df},
        save_path=os.path.join(figures_dir, "eda_splits.png")
    )
    plot_entity_distributions(
        train_df,
        save_path=os.path.join(figures_dir, "eda_entity_dist.png")
    )
    plot_sentence_length_distribution(
        train_df,
        save_path=os.path.join(figures_dir, "eda_sentence_lengths.png")
    )
    print(f"  Saved EDA figures to {figures_dir}")

    # 3. Feature Extraction
    print("\n[Step 3/6] Extracting Linguistic Features...")
    t0 = time.time()
    X_train_dicts, y_train, train_lens = extract_features_from_dataframe(train_df)
    print(f"  Extracted {len(X_train_dicts):,} train token features in {time.time()-t0:.2f}s")

    t0 = time.time()
    X_val_dicts, y_val, val_lens = extract_features_from_dataframe(val_df)
    print(f"  Extracted {len(X_val_dicts):,} val token features in {time.time()-t0:.2f}s")

    t0 = time.time()
    X_test_dicts, y_test, test_lens = extract_features_from_dataframe(test_df)
    print(f"  Extracted {len(X_test_dicts):,} test token features in {time.time()-t0:.2f}s")

    # 4. Model Training & Hyperparameter Tuning
    print("\n[Step 4/6] Training SVM Model & Selecting Regularization Parameter C...")
    model = SVMNERModel(C=0.5, max_iter=2000, random_state=42, dual=False)
    
    # We evaluate candidates C in [0.05, 0.1, 0.5, 1.0]
    tuning_res = model.tune_hyperparameters(
        X_train_dicts, y_train,
        X_val_dicts, y_val,
        c_candidates=[0.05, 0.1, 0.5, 1.0]
    )

    model_path = os.path.join(output_dir, "svm_ner_model.joblib")
    model.save(model_path)

    # 5. Model Evaluation
    print("\n[Step 5/6] Evaluating Model on Unseen Test Set...")
    y_test_pred = model.predict(X_test_dicts)
    y_test_pred_list = list(y_test_pred)

    # Token-level metrics
    token_metrics = compute_token_level_metrics(y_test, y_test_pred_list)
    print("\n" + "=" * 35 + " TOKEN-LEVEL REPORT " + "=" * 35)
    print(token_metrics['classification_report_text'])
    print(f"Overall Test Accuracy:    {token_metrics['accuracy']:.4f}")
    print(f"Overall Macro Precision:  {token_metrics['macro_precision']:.4f}")
    print(f"Overall Macro Recall:     {token_metrics['macro_recall']:.4f}")
    print(f"Overall Macro F1-Score:   {token_metrics['macro_f1']:.4f}")
    print(f"Overall Weighted F1-Score:{token_metrics['weighted_f1']:.4f}")

    # Entity-level metrics (Seqeval)
    entity_metrics = compute_entity_level_metrics(y_test, y_test_pred_list, test_lens)
    print("\n" + "=" * 35 + " ENTITY-LEVEL REPORT (SEQEVAL) " + "=" * 35)
    print(entity_metrics['entity_report_text'])
    print(f"Entity-Level Micro Precision: {entity_metrics['entity_precision']:.4f}")
    print(f"Entity-Level Micro Recall:    {entity_metrics['entity_recall']:.4f}")
    print(f"Entity-Level Micro F1-Score:  {entity_metrics['entity_f1']:.4f}")

    # Plot evaluation charts
    plot_confusion_matrix(
        token_metrics['confusion_matrix'],
        token_metrics['labels'],
        normalize=True,
        save_path=os.path.join(figures_dir, "confusion_matrix.png")
    )
    plot_per_class_metrics(
        token_metrics['classification_report_dict'],
        save_path=os.path.join(figures_dir, "per_class_metrics.png")
    )

    # 6. Detailed Error Analysis
    print("\n[Step 6/6] Performing Error Analysis...")
    error_analysis = analyze_errors(test_df, y_test, y_test_pred_list, test_lens, max_examples=10)
    plot_error_type_distribution(
        error_analysis['error_counts_by_category'],
        save_path=os.path.join(figures_dir, "error_breakdown.png")
    )

    print("\n--- Error Breakdown by Category ---")
    for category, count in error_analysis['error_counts_by_category'].items():
        print(f"  {category:<32}: {count:,}")

    print("\n--- Sample Misclassifications ---")
    for i, item in enumerate(error_analysis['sampled_error_sentences'][:3]):
        print(f"\n[Error Case #{i+1} - Sentence Index {item['sentence_index']}]")
        sentence_str = " ".join(item['tokens'])
        print(f"  Sentence: \"{sentence_str}\"")
        discrepancies = [t for t in item['token_comparisons'] if not t['is_correct']]
        for disc in discrepancies:
            print(f"    Token '{disc['token']}': Ground Truth = {disc['true_tag']} | Predicted = {disc['pred_tag']}")

    # Save summary metrics JSON
    summary_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'hyperparameters': {
            'best_C': tuning_res['best_C'],
            'tuning_history': tuning_res['tuning_history'],
            'max_iter': model.max_iter,
            'dual': model.dual,
            'random_state': model.random_state
        },
        'token_level_test_metrics': {
            'accuracy': float(token_metrics['accuracy']),
            'macro_precision': float(token_metrics['macro_precision']),
            'macro_recall': float(token_metrics['macro_recall']),
            'macro_f1': float(token_metrics['macro_f1']),
            'weighted_precision': float(token_metrics['weighted_precision']),
            'weighted_recall': float(token_metrics['weighted_recall']),
            'weighted_f1': float(token_metrics['weighted_f1']),
            'per_class_f1': {
                k: float(v['f1-score'])
                for k, v in token_metrics['classification_report_dict'].items()
                if isinstance(v, dict) and 'f1-score' in v
            }
        },
        'entity_level_test_metrics': {
            'precision': float(entity_metrics['entity_precision']),
            'recall': float(entity_metrics['entity_recall']),
            'f1_score': float(entity_metrics['entity_f1'])
        },
        'error_breakdown': error_analysis['error_counts_by_category']
    }

    metrics_file = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[Pipeline] Execution successfully completed. Metrics serialized to {metrics_file}.")


if __name__ == '__main__':
    main()
