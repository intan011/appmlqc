"""
be_qc_lib_saved.py
Auto-generated safe library for predictions.
FIXED VERSION - works with both folder structures
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def load_target_models(out_dir, target):
    """
    Try to load models from either:
    1. Subdirectory structure: out_dir/target/meta_target.json
    2. Flat structure: out_dir/meta_target.json
    """
    # Try subdirectory structure first
    tdir = os.path.join(out_dir, target)
    meta_path = os.path.join(tdir, f"meta_{target}.json")
    
    # If subdirectory doesn't exist, try flat structure
    if not os.path.exists(meta_path):
        tdir = out_dir
        meta_path = os.path.join(out_dir, f"meta_{target}.json")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata for target '{target}' not found at: {meta_path}")
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Load preprocessor and models (paths in meta might be absolute or relative)
    preproc_path = meta["preproc"]
    if not os.path.isabs(preproc_path):
        preproc_path = os.path.join(tdir, os.path.basename(preproc_path))
    
    lgb_lower_path = meta["lgb_lower"]
    if not os.path.isabs(lgb_lower_path):
        lgb_lower_path = os.path.join(tdir, os.path.basename(lgb_lower_path))
    
    lgb_median_path = meta["lgb_median"]
    if not os.path.isabs(lgb_median_path):
        lgb_median_path = os.path.join(tdir, os.path.basename(lgb_median_path))
    
    lgb_upper_path = meta["lgb_upper"]
    if not os.path.isabs(lgb_upper_path):
        lgb_upper_path = os.path.join(tdir, os.path.basename(lgb_upper_path))
    
    preproc = joblib.load(preproc_path)
    m_low = lgb.Booster(model_file=lgb_lower_path)
    m_med = lgb.Booster(model_file=lgb_median_path)
    m_up  = lgb.Booster(model_file=lgb_upper_path)
    
    return preproc, m_low, m_med, m_up, meta

def predict_new(df_new, out_dir="be_qc_models", targets=None):
    if targets is None:
        targets = ["OUTPUT","INPUT","NILAI_DITAMBAH","GAJI_UPAH","JUMLAH_PEKERJA"]
    
    df = df_new.copy()
    
    # Derived features (only if columns present)
    if "JUMLAH_PEKERJA" in df.columns:
        df["log_workers"] = np.log1p(df["JUMLAH_PEKERJA"])
    if {"HARTA_TETAP","JUMLAH_PEKERJA"}.issubset(df.columns):
        df["asset_per_worker"] = (df["HARTA_TETAP"] / df["JUMLAH_PEKERJA"]).replace([np.inf, -np.inf], np.nan)
    if {"GAJI_UPAH","JUMLAH_PEKERJA"}.issubset(df.columns):
        df["wage_per_worker"] = (df["GAJI_UPAH"] / df["JUMLAH_PEKERJA"]).replace([np.inf, -np.inf], np.nan)
    
    df_out = df.copy()
    
    for t in targets:
        try:
            print(f"Loading models for target: {t}")
            preproc, m_low, m_med, m_up, meta = load_target_models(out_dir, t)
            print(f"✓ Successfully loaded models for {t}")
        except Exception as e:
            # artifact missing; skip this target
            print(f"✗ Warning: Could not load models for {t}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
        
        feats_num = meta.get("features_num", [])
        feats_cat = meta.get("features_cat", [])
        
        # Safe construction of X_raw
        X_raw = pd.DataFrame()
        for c in feats_num:
            if c in df_out.columns:
                X_raw[c] = to_num(df_out[c])
            else:
                X_raw[c] = np.nan
        for c in feats_cat:
            if c in df_out.columns:
                X_raw[c] = df_out[c].astype(str)
            else:
                X_raw[c] = "__MISS__"
        
        # Transform & predict
        X = preproc.transform(X_raw)
        df_out[f"{t}_PRED_MED"] = m_med.predict(X)
        df_out[f"{t}_PRED_LOW"] = m_low.predict(X)
        df_out[f"{t}_PRED_UP"]  = m_up.predict(X)
        print(f"✓ Predictions generated for {t}")
        
        # Flag if reported outside predicted interval (if reported exists)
        if t in df_out.columns:
            df_out[f"{t}_FLAG"] = ~df_out[t].between(df_out[f"{t}_PRED_LOW"], df_out[f"{t}_PRED_UP"])
            df_out.loc[df_out[t].isna(), f"{t}_FLAG"] = False
        else:
            df_out[f"{t}_FLAG"] = False
    
    # Composite flag
    flag_cols = [c for c in df_out.columns if c.endswith("_FLAG")]
    df_out["FLAG_FINAL"] = df_out[flag_cols].any(axis=1) if flag_cols else False
    
    return df_out

def predict_single(record_dict, out_dir="be_qc_models", targets=None):
    df = pd.DataFrame([record_dict])
    df_pred = predict_new(df, out_dir=out_dir, targets=targets)
    return df_pred.iloc[0].to_dict()
