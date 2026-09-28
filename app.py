"""FastAPI Web Application for CoNLL-2003 Named Entity Recognition using Support Vector Machines.

Author: Muhammad Hammad

Hosts an interactive UI on localhost:8000 for live text prediction,
interactive metrics dashboards, test set exploration, and error analysis.
"""

import json
import os
import re
import string
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.features import extract_token_features
from src.data_loader import POS_TAG_NAMES, NER_TAG_NAMES, load_conll2003
from src.model import ensure_int32_indices

app = FastAPI(
    title="CoNLL-2003 Named Entity Recognition - SVM Web App",
    description="Interactive Web Application and REST API by Muhammad Hammad",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model artifacts
MODEL_PATH = "outputs/svm_ner_model.joblib"
METRICS_PATH = "outputs/metrics_summary.json"

model_bundle = None
metrics_data = None
test_samples_cache = []

try:
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)
        print(f"[WebApp] Successfully loaded model from {MODEL_PATH}")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    
    # Load test split automatically (supports .txt, .parquet, and Kaggle inputs)
    _, _, test_df = load_conll2003()
    if not test_df.empty:
        for _, row in test_df.iterrows():
            if any(tag != 0 for tag in row['ner_tags']) and len(row['tokens']) > 5:
                test_samples_cache.append({
                    'id': str(row['id']),
                    'tokens': [str(t) for t in row['tokens']],
                    'pos_tags': [int(p) for p in row['pos_tags']],
                    'ner_tags': [str(NER_TAG_NAMES[int(t)]) for t in row['ner_tags']]
                })
                if len(test_samples_cache) >= 30:
                    break
        print(f"[WebApp] Cached {len(test_samples_cache)} test set examples.")
except Exception as e:
    print(f"[WebApp] Warning during startup: {e}")


# Rule-based POS tagger heuristic for custom user text
COMMON_POS = {
    'the': 'DT', 'a': 'DT', 'an': 'DT', 'this': 'DT', 'that': 'DT', 'these': 'DT', 'those': 'DT',
    'in': 'IN', 'on': 'IN', 'at': 'IN', 'to': 'TO', 'of': 'IN', 'for': 'IN', 'with': 'IN',
    'by': 'IN', 'from': 'IN', 'about': 'IN', 'into': 'IN', 'through': 'IN', 'after': 'IN',
    'before': 'IN', 'between': 'IN', 'under': 'IN', 'over': 'IN',
    'and': 'CC', 'but': 'CC', 'or': 'CC', 'nor': 'CC', 'yet': 'CC', 'so': 'CC',
    'is': 'VBZ', 'was': 'VBD', 'are': 'VBP', 'were': 'VBD', 'be': 'VB', 'been': 'VBN', 'being': 'VBG',
    'have': 'VBP', 'has': 'VBZ', 'had': 'VBD', 'do': 'VBP', 'does': 'VBZ', 'did': 'VBD',
    'will': 'MD', 'would': 'MD', 'shall': 'MD', 'should': 'MD', 'can': 'MD', 'could': 'MD', 'may': 'MD', 'might': 'MD',
    'said': 'VBD', 'says': 'VBZ', 'told': 'VBD', 'reported': 'VBD', 'announced': 'VBD', 'met': 'VBD',
    'he': 'PRP', 'she': 'PRP', 'it': 'PRP', 'they': 'PRP', 'we': 'PRP', 'i': 'PRP', 'you': 'PRP',
    'his': 'PRP$', 'her': 'PRP$', 'its': 'PRP$', 'their': 'PRP$', 'our': 'PRP$', 'my': 'PRP$',
    'not': 'RB', 'never': 'RB', 'also': 'RB', 'well': 'RB', 'very': 'RB', 'too': 'RB',
    'mr.': 'NNP', 'mr': 'NNP', 'mrs.': 'NNP', 'mrs': 'NNP', 'ms.': 'NNP', 'dr.': 'NNP', 'prof.': 'NNP',
    'president': 'NNP', 'minister': 'NNP', 'prime': 'NNP', 'secretary': 'NNP', 'chancellor': 'NNP',
    'first': 'JJ', 'second': 'JJ', 'third': 'JJ', 'new': 'JJ', 'old': 'JJ', 'last': 'JJ', 'next': 'JJ'
}

POS_NAME_TO_ID = {name: i for i, name in enumerate(POS_TAG_NAMES)}


def estimate_pos_tag(token: str, index: int, total_tokens: int) -> int:
    """Heuristic Penn Treebank POS tag estimator for arbitrary user input."""
    lower = token.lower()
    if lower in COMMON_POS:
        tag_name = COMMON_POS[lower]
        return POS_NAME_TO_ID.get(tag_name, 21) # default NN

    if token.isdigit():
        return POS_NAME_TO_ID.get('CD', 11)

    if any(c in string.punctuation for c in token) and len(token) == 1:
        return POS_NAME_TO_ID.get(token, 7) # '.'

    if token[0].isupper() and (index > 0 or token.lower() not in ['the', 'a', 'in', 'it', 'on', 'at']):
        return POS_NAME_TO_ID.get('NNP', 22)

    if lower.endswith('ly'):
        return POS_NAME_TO_ID.get('RB', 30)
    if lower.endswith('ing'):
        return POS_NAME_TO_ID.get('VBG', 39)
    if lower.endswith('ed'):
        return POS_NAME_TO_ID.get('VBD', 38)
    if lower.endswith('s') and len(lower) > 3:
        return POS_NAME_TO_ID.get('NNS', 24)

    return POS_NAME_TO_ID.get('NN', 21)


def tokenize_text(text: str) -> List[str]:
    """Tokenize raw text into words and punctuation marks matching CoNLL style."""
    pattern = r"[\w]+|[^\w\s]"
    tokens = re.findall(pattern, text)
    return tokens


class PredictRequest(BaseModel):
    text: str


class TokenPrediction(BaseModel):
    token: str
    pos: str
    predicted_tag: str
    entity_type: str
    is_entity: bool


class PredictResponse(BaseModel):
    tokens: List[str]
    predictions: List[TokenPrediction]
    formatted_html: str
    detected_entities: List[Dict[str, Any]]


@app.post("/api/predict", response_model=PredictResponse)
def predict_entities(payload: PredictRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Trained model is not loaded.")

    tokens = tokenize_text(payload.text)
    if not tokens:
        raise HTTPException(status_code=400, detail="No valid tokens found.")

    pos_tag_ids = [estimate_pos_tag(t, i, len(tokens)) for i, t in enumerate(tokens)]
    
    # Extract features
    features = [extract_token_features(tokens, pos_tag_ids, i) for i in range(len(tokens))]
    
    # Transform with model bundle
    vec = model_bundle['vectorizer']
    clf = model_bundle['classifier']
    
    X_sparse = vec.transform(features)
    X_sparse = ensure_int32_indices(X_sparse)
    
    y_pred = clf.predict(X_sparse)
    
    # Reconstruct entities into chunks
    predictions = []
    detected_entities = []
    
    current_entity = None
    
    for i, (tok, tag) in enumerate(zip(tokens, y_pred)):
        pos_str = POS_TAG_NAMES[pos_tag_ids[i]] if 0 <= pos_tag_ids[i] < len(POS_TAG_NAMES) else "UNK"
        
        ent_type = "O"
        is_ent = (tag != "O")
        if is_ent:
            ent_type = tag[2:] # 'PER', 'ORG', 'LOC', 'MISC'
            
        predictions.append(TokenPrediction(
            token=tok,
            pos=pos_str,
            predicted_tag=tag,
            entity_type=ent_type,
            is_entity=is_ent
        ))
        
        # Entity span tracking
        if tag.startswith("B-"):
            if current_entity:
                detected_entities.append(current_entity)
            current_entity = {
                'text': tok,
                'type': ent_type,
                'start_token': i,
                'end_token': i,
                'tokens': [tok]
            }
        elif tag.startswith("I-") and current_entity and current_entity['type'] == ent_type:
            current_entity['text'] += " " + tok
            current_entity['end_token'] = i
            current_entity['tokens'].append(tok)
        else:
            if current_entity:
                detected_entities.append(current_entity)
                current_entity = None
                
    if current_entity:
        detected_entities.append(current_entity)

    # Build highlighted HTML markup
    html_parts = []
    for pred in predictions:
        tok = pred.token
        if pred.is_entity:
            cls_name = f"entity-tag tag-{pred.entity_type.lower()}"
            html_parts.append(
                f'<span class="{cls_name}" title="{pred.predicted_tag} (POS: {pred.pos})">'
                f'{tok}<span class="badge">{pred.entity_type}</span></span>'
            )
        else:
            html_parts.append(f'<span class="token-normal">{tok}</span>')

    formatted_html = " ".join(html_parts)
    # Fix punctuation spacing in HTML
    formatted_html = re.sub(r'\s+([,.:;!?\'])', r'\1', formatted_html)

    return PredictResponse(
        tokens=tokens,
        predictions=predictions,
        formatted_html=formatted_html,
        detected_entities=detected_entities
    )


@app.get("/api/metrics")
def get_metrics():
    if metrics_data:
        return metrics_data
    return JSONResponse(status_code=404, content={"message": "Metrics not found."})


@app.get("/api/test-samples")
def get_test_samples():
    if test_samples_cache:
        return test_samples_cache
    return JSONResponse(status_code=404, content={"message": "No test samples available."})


@app.get("/api/figures/{figure_name}")
def get_figure(figure_name: str):
    allowed_figures = [
        "eda_splits.png",
        "eda_entity_dist.png",
        "eda_sentence_lengths.png",
        "confusion_matrix.png",
        "per_class_metrics.png",
        "error_breakdown.png"
    ]
    if figure_name not in allowed_figures:
        raise HTTPException(status_code=404, detail="Figure not found.")
    for candidate_dir in ["figures", os.path.join("outputs", "figures")]:
        path = os.path.join(candidate_dir, figure_name)
        if os.path.exists(path):
            return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Figure file missing.")


# Serve static frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>CoNLL-2003 NER SVM Web Application</h1><p>Frontend static files loading...</p>"


if __name__ == '__main__':
    import uvicorn
    print("\n" + "="*70)
    print(" Starting CoNLL-2003 NER SVM Web Application on http://localhost:8000")
    print("="*70 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
