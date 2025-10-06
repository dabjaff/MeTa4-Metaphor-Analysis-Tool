from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional
from . import config
from .utils import require_nonempty_df
__all__ = ['normalize_dataframe', 'ensure_normalized', 'apply_mipvu_filters', 'mipvu_counts', 'prepare_analysis']

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    CANON = {'File_ID': 'file_id', 'Genre': 'genre', 'Sentence_ID': 'sentence_id', 'Original_Word': 'word', 'Lemma': 'lemma', 'POS': 'pos', 'Metaphor': 'metaphor_function', 'Type': 'type', 'Subtype': 'subtype', 'MFlag': 'mflag', 'xml:id': 'xml_id', 'corresp': 'corresp'}
    col_map = {c: CANON.get(c, c.lower()) for c in df.columns}
    df = df.rename(columns=col_map).copy()
    for c in ['lemma', 'pos', 'metaphor_function', 'type', 'subtype', 'mflag', 'genre']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    if 'sentence_id' in df.columns:
        df['sentence_id'] = df['sentence_id'].astype(str)
    return df

def ensure_normalized(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if getattr(df, 'attrs', None) and df.attrs.get('normalized', False):
            return df
    except Exception:
        pass
    df_n = normalize_dataframe(df)
    try:
        df_n.attrs['normalized'] = True
    except Exception:
        pass
    return df_n

def apply_mipvu_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = ensure_normalized(df)
    keep = ~df['subtype'].str.upper().isin({'DFMA', 'DFMA_PUNCT'}) & (df['metaphor_function'].str.upper() != 'DFMA')
    df_f = df.loc[keep].copy()
    df_f.attrs = df.attrs.copy()
    denom_mask = ~df_f['pos'].astype(str).str.lower().str.startswith('pun')
    typ = df_f['type']
    sub = df_f['subtype'].str.upper()
    flag = df_f['mflag']
    mrw_mask = df_f['metaphor_function'].astype(str).str.lower().eq('mrw')
    mrw_mask &= ~(typ.isin({'lex', 'morph', 'phrase'}) | (flag == 'mflag') | sub.isin({'DFMA', 'UNKNOWN'}))
    try:
        if str(df_f.attrs.get('corpus', '')).lower() == 'vuamc':
            bad = df_f['genre'].astype(str).str.lower().eq('news') & df_f['lemma'].astype(str).str.lower().eq('of')
            corrected = bad.sum()
            if corrected > 0:
                config.logger.debug(f"Corrected {corrected} 'of' MRWs in News")
            mrw_mask &= ~bad
    except Exception:
        pass
    return (df_f, denom_mask, mrw_mask)

@require_nonempty_df
def mipvu_counts(df: pd.DataFrame) -> Tuple[int, int]:
    _, den, mrw = apply_mipvu_filters(df)
    return (int((mrw & den).sum()), int(den.sum()))

def prepare_analysis(df: pd.DataFrame, lemma: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, int]:
    lemma = lemma.lower()
    if not lemma:
        raise ValueError('Empty lemma provided')
    df_f, den, mrw = apply_mipvu_filters(df)
    hit_count = (df_f['lemma'] == lemma).sum()
    if hit_count == 0:
        config.logger.warning(f"This lemma '{lemma}' is not found in the corpus")
    return (df_f, den, mrw, hit_count)
