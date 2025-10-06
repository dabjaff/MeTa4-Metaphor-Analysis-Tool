from __future__ import annotations
import os
import re
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from . import config
from .utils import get_timestamp, format_table, require_nonempty_df, _norm_id
from .utils import _merge_pipe
from .mipvu import ensure_normalized, apply_mipvu_filters, mipvu_counts, prepare_analysis
__all__ = ['extract_context', 'save_concordance_dual', 'single_lemma_report_classic', 'save_collocations_dual', 'find_lemma_regex_hits', 'export_pattern_kwic', 'export_flat_vuamc_csv', 'export_mrw_list', 'export_mrw_list_by_pos']

def extract_context(df: pd.DataFrame, indices: pd.Index, window: int, direction: str, mrw_mask: pd.Series) -> List[str]:
    out: List[str] = []
    words = df.get('word', pd.Series([''] * len(df), index=df.index)).astype(str).tolist()
    sids = df.get('sentence_id', pd.Series(['_one'] * len(df), index=df.index)).tolist()
    if isinstance(mrw_mask, pd.Series):
        mrw_arr = mrw_mask.reindex(df.index, fill_value=False).to_numpy()
    else:
        mrw_arr = pd.Series([False] * len(df), index=df.index).to_numpy()
    n = len(df)
    if isinstance(indices, (int,)):
        idx_iter = [indices]
    else:
        idx_iter = list(indices)
    for pos in idx_iter:
        sid = sids[pos]
        start_pos = pos
        while start_pos > 0 and sids[start_pos - 1] == sid:
            start_pos -= 1
        end_pos = pos + 1
        while end_pos < n and sids[end_pos] == sid:
            end_pos += 1
        if direction == 'left':
            left_start = max(start_pos, pos - window)
            rel_idx = list(range(left_start, pos))
            tokens = [words[j] + '_Ω' if mrw_arr[j] else words[j] for j in rel_idx]
            if not rel_idx or rel_idx[0] == start_pos:
                tokens = ['[SENT_START]'] + tokens
            out.append(' '.join(tokens))
        else:
            right_end = min(end_pos, pos + 1 + window)
            rel_idx = list(range(pos + 1, right_end))
            tokens = [words[j] + '_Ω' if mrw_arr[j] else words[j] for j in rel_idx]
            if not rel_idx or rel_idx[-1] == end_pos - 1:
                tokens = tokens + ['[SENT_END]']
            out.append(' '.join(tokens))
    return out

def save_concordance_dual(df: pd.DataFrame, lemma: str, outdir: str, window: Optional[int]=None, stamp: Optional[str]=None) -> None:
    if window is None:
        window = config.CONFIG.get('DEFAULT_WINDOW', 15)
    lemma = str(lemma or '').lower().strip()
    if not lemma:
        raise ValueError('Empty lemma provided')
    df_f, _, mrw_mask = apply_mipvu_filters(df)
    indices = df_f.index[df_f['lemma'] == lemma]
    if indices.empty:
        return
    is_mrw = mrw_mask.loc[indices]
    for mrw_only, suffix in [(True, '_MRW'), (False, '_NONMRW')]:
        sub_indices = indices[is_mrw] if mrw_only else indices[~is_mrw]
        if sub_indices.empty:
            continue
        lefts = extract_context(df_f, sub_indices, window, 'left', mrw_mask)
        rights = extract_context(df_f, sub_indices, window, 'right', mrw_mask)
        nodes = df_f.loc[sub_indices, ['word', 'pos']].apply(lambda row: f"{row['word']}_{row['pos']}", axis=1)
        rel_mrw = ['MRW' if mrw else 'non-MRW' for mrw in is_mrw.loc[sub_indices]]
        data = pd.DataFrame({'Left': lefts, 'Node': nodes, 'relation_mrw': rel_mrw, 'Right': rights, 'lemma': lemma, 'genre': df_f.loc[sub_indices, 'genre'], 'sentence_id': df_f.loc[sub_indices, 'sentence_id'], 'metaphor_function': df_f.loc[sub_indices, 'metaphor_function']})
        fn = f'{lemma}_KWIC{suffix}_{(stamp if stamp else get_timestamp())}.csv'
        os.makedirs(outdir, exist_ok=True)
        data.to_csv(os.path.join(outdir, fn), index=False, encoding='utf-8-sig')

@require_nonempty_df
def single_lemma_report_classic(df: pd.DataFrame, lemma: str, whole_df: pd.DataFrame) -> str:
    lemma = lemma.lower()
    if not lemma:
        raise ValueError('Empty lemma provided')
    raw_occ = (df['lemma'] == lemma).sum()
    df_f, den_mask, mrw_mask, lu_occ = prepare_analysis(df, lemma)
    if lu_occ == 0:
        similar = whole_df['lemma'].unique()
        from difflib import get_close_matches
        suggestions = get_close_matches(lemma, similar, n=3, cutoff=0.6)
        if suggestions:
            return f"No lemma '{lemma}' found. Did you mean: {', '.join(suggestions)}?"
        return f"No lemma '{lemma}' found in the corpus."
    num_mrw = (mrw_mask & den_mask & (df_f['lemma'] == lemma)).sum()
    total_lu = den_mask.sum()
    density = num_mrw / total_lu * config.CONFIG.get('PER_1000', 1000) if total_lu else 0
    share = lu_occ / total_lu * 100 if total_lu else 0
    report = f"--- Analysis for '{lemma}' ---\n"
    report += 'Whole corpus:\n'
    report += f'  raw_occurrences (all rows): {raw_occ}\n'
    report += f'  LU_occurrences (denominator): {lu_occ}\n'
    report += f'  MRW_occurrences (strict): {num_mrw}\n'
    report += f'  %MRW_within_lemma (LU-based): {(num_mrw / lu_occ * 100 if lu_occ else 0):.2f}%\n'
    report += f'  MRW_density_per_1000_LU: {density:.3f}\n'
    report += f'  lemma_share_of_corpus_LU (%): {share:.3f}\n'
    if 'genre' in df_f.columns:
        report += 'Per genre:\n'
        headers = ['Genre', 'LUs', 'LU_occ', 'MRWs', '%MRW']
        rows = []
        for g, sub in df_f.groupby('genre'):
            sub_total_lu = den_mask[sub.index].sum()
            sub_lu_occ = (sub['lemma'] == lemma).sum()
            sub_mrw = ((mrw_mask & den_mask)[sub.index] & (sub['lemma'] == lemma)).sum()
            pct = sub_mrw / sub_lu_occ * 100 if sub_lu_occ else 0
            rows.append([g.capitalize(), sub_total_lu, sub_lu_occ, sub_mrw, f'{pct:.2f}%'])
        report += format_table(headers, rows)
    return report

@require_nonempty_df
def save_collocations_dual(df: pd.DataFrame, lemma: str, outdir: str, window: Optional[int]=None, stamp: Optional[str]=None) -> None:
    df = ensure_normalized(df)
    if window is None:
        window = config.CONFIG.get('COLLOCATION_WINDOW', config.CONFIG.get('DEFAULT_WINDOW', 5))
    if window < 0:
        raise ValueError('Window must be non‑negative')
    lemma = str(lemma or '').lower().strip()
    if not lemma:
        raise ValueError('Empty lemma provided')
    hits = df.index[df['lemma'].astype(str).str.lower() == lemma].tolist()
    os.makedirs(outdir, exist_ok=True)
    fn = f'collocations_{lemma}_w{window}' + (f'_{stamp}' if stamp else '') + '.csv'
    if not hits:
        pd.DataFrame(columns=['lemma', 'collocate', 'pos', 'side', 'distance', 'count']).to_csv(os.path.join(outdir, fn), index=False, encoding='utf-8-sig')
        return
    rows: List[Tuple[str, str, str, str, int, int]] = []
    sent_groups = df.groupby('sentence_id', sort=False)
    for _, sent in sent_groups:
        lemmas = sent['lemma'].astype(str).str.lower().tolist()
        poss = sent['pos'].astype(str).str.lower().tolist() if 'pos' in sent.columns else [''] * len(lemmas)
        for pos_i, lm in enumerate(lemmas):
            if lm != lemma:
                continue
            L = max(0, pos_i - window)
            for k in range(L, pos_i):
                dist = pos_i - k
                rows.append((lemma, lemmas[k], poss[k], 'L', dist, 1))
            R = min(len(lemmas), pos_i + window + 1)
            for k in range(pos_i + 1, R):
                dist = k - pos_i
                rows.append((lemma, lemmas[k], poss[k], 'R', dist, 1))
    out = pd.DataFrame(rows, columns=['lemma', 'collocate', 'pos', 'side', 'distance', 'count'])
    if not out.empty:
        out = out.groupby(['lemma', 'collocate', 'pos', 'side', 'distance'], as_index=False)['count'].sum().sort_values(['side', 'distance', 'count'], ascending=[True, True, False])
    out.to_csv(os.path.join(outdir, fn), index=False, encoding='utf-8-sig')

def find_lemma_regex_hits(df: pd.DataFrame, q: str) -> List[int]:
    q = str(q or '')
    if q.startswith('re:'):
        pat = re.compile(q[3:], re.IGNORECASE)
        return df.index[df['lemma'].str.match(pat, na=False)].tolist()
    pat_str = '^' + re.escape(q).replace('\\*', '.*').replace('?', '.') + '$'
    return df.index[df['lemma'].str.match(pat_str, case=False, na=False)].tolist()

def export_pattern_kwic(df: pd.DataFrame, spans: List[Tuple[str, List[int]]], path: str, q: str, raw_mode: bool=False, window: Optional[int]=None, mrw_only: bool=False) -> None:
    if window is None:
        window = config.CONFIG.get('DEFAULT_WINDOW', 15)
    original_len = sum((len(ix) for _, ix in spans))
    if raw_mode:
        df_eff = df
        mrw_mask = pd.Series(dtype=bool)
        kept_spans = spans
        config.logger.info(f"Pattern '{q}': {original_len} raw matches (no MIPVU applied)")
    else:
        df_f, _, mrw_mask = apply_mipvu_filters(df)
        kept_spans: List[Tuple[str, List[int]]] = []
        kept_tokens = 0
        kept_mrw = 0
        for sid, indices in spans:
            valid = [idx for idx in indices if idx in df_f.index]
            if mrw_only:
                valid = [idx for idx in valid if mrw_mask.get(idx, False)]
            if valid:
                kept_spans.append((sid, valid))
                kept_tokens += len(valid)
                if not mrw_only:
                    kept_mrw += sum((1 for idx in valid if mrw_mask.get(idx, False)))
                else:
                    kept_mrw += len(valid)
        config.logger.info(f"Pattern '{q}': {original_len} raw tokens; {kept_tokens} after MIPVU" + (' (MRW‑only)' if mrw_only else f' ({kept_mrw} MRW among kept)'))
        df_eff = df_f
    rows: List[Dict[str, Any]] = []
    for sid, indices in kept_spans:
        for idx in indices:
            left = extract_context(df_eff, pd.Index([idx]), window, 'left', mrw_mask if not raw_mode else pd.Series(dtype=bool))
            right = extract_context(df_eff, pd.Index([idx]), window, 'right', mrw_mask if not raw_mode else pd.Series(dtype=bool))
            node = df_eff.at[idx, 'word'] + '_' + df_eff.at[idx, 'pos']
            rel_mrw = 'MRW' if not raw_mode and mrw_mask.get(idx, False) else 'raw' if raw_mode else 'non-MRW'
            rows.append({'sentence_id': sid, 'Left': left[0], 'Node': node, 'relation_mrw': rel_mrw, 'Right': right[0], 'query': q})
    out_df = pd.DataFrame(rows, columns=['sentence_id', 'Left', 'Node', 'relation_mrw', 'Right', 'query'])
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out_df.to_csv(path, index=False, encoding='utf-8-sig')

def export_flat_vuamc_csv(df: pd.DataFrame, out_path: str) -> None:
    df_f = ensure_normalized(df).copy()
    required = ['file_id', 'genre', 'sentence_id', 'word', 'lemma', 'pos', 'metaphor_function', 'type', 'subtype', 'mflag', 'xml_id', 'corresp']
    for col in required:
        if col not in df_f.columns:
            df_f[col] = ''
    ordered = ['file_id', 'genre', 'sentence_id', 'word', 'lemma', 'pos', 'metaphor_function', 'type', 'subtype', 'mflag', 'xml_id', 'corresp']
    col_title = {'file_id': 'File_ID', 'genre': 'Genre', 'sentence_id': 'Sentence_ID', 'word': 'Original_Word', 'lemma': 'Lemma', 'pos': 'POS', 'metaphor_function': 'Metaphor', 'type': 'Type', 'subtype': 'Subtype', 'mflag': 'MFlag', 'xml_id': 'xml:id', 'corresp': 'corresp'}
    out = df_f.loc[:, ordered].copy()
    for c in out.columns:
        try:
            out[c] = out[c].astype(str)
        except Exception:
            out[c] = out[c].map(lambda x: '' if x is None else str(x))
        out[c] = out[c].replace('nan', '')
    out = out.rename(columns=col_title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding='utf-8-sig')

def export_mrw_list(df: pd.DataFrame, out_path: str) -> None:
    df_f, _, mrw = apply_mipvu_filters(df)
    if 'word' not in df_f.columns:
        raise ValueError("Expected 'word' column in DataFrame.")
    meta_all = _build_metaphor_all_column(df_f)
    out = pd.DataFrame({'Original Word': df_f.loc[mrw, 'word'].astype(str), 'Lemma': df_f.loc[mrw, 'lemma'].astype(str) if 'lemma' in df_f.columns else '', 'Metaphor (all)': meta_all.loc[mrw] if len(meta_all) == len(df_f) else '', 'Genre': df_f.loc[mrw, 'genre'].astype(str) if 'genre' in df_f.columns else '', 'Sentence ID': df_f.loc[mrw, 'sentence_id'].astype(str) if 'sentence_id' in df_f.columns else ''})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding='utf-8-sig')

def export_mrw_list_by_pos(df: pd.DataFrame, pos_query: str, out_path: str) -> None:
    df_f, _, mrw = apply_mipvu_filters(df)
    if 'pos' not in df_f.columns:
        raise ValueError("Expected 'pos' column in DataFrame.")
    pq = (pos_query or '').strip().lower()

    def pos_match(s: str) -> bool:
        toks = [t.strip() for t in str(s).lower().split('+')]
        return pq in toks
    pos_mask = df_f['pos'].apply(pos_match)
    keep = mrw & pos_mask
    meta_all = _build_metaphor_all_column(df_f)
    out = pd.DataFrame({'Original Word': df_f.loc[keep, 'word'].astype(str) if 'word' in df_f.columns else '', 'Lemma': df_f.loc[keep, 'lemma'].astype(str) if 'lemma' in df_f.columns else '', 'Metaphor (all)': meta_all.loc[keep] if len(meta_all) == len(df_f) else '', 'Genre': df_f.loc[keep, 'genre'].astype(str) if 'genre' in df_f.columns else '', 'Sentence ID': df_f.loc[keep, 'sentence_id'].astype(str) if 'sentence_id' in df_f.columns else ''})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding='utf-8-sig')

def _build_metaphor_all_column(df: pd.DataFrame) -> pd.Series:
    parts: List[pd.Series] = []
    for col in ['metaphor_function', 'type', 'subtype', 'mflag']:
        if col in df.columns:
            parts.append(df[col].astype(str).str.strip())
        else:
            parts.append(pd.Series([''] * len(df)))
    combo = parts[0]
    for p in parts[1:]:
        combo = combo.str.cat(p, sep=', ', na_rep='')
    combo = combo.str.replace('\\s*,\\s*,+', ', ', regex=True)
    combo = combo.str.replace('^(,\\s*)+|(,\\s*)+$', '', regex=True)
    combo = combo.str.replace('\\s{2,}', ' ', regex=True).str.strip()
    return combo
