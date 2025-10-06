from __future__ import annotations
import re
import os
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from . import config
from .utils import _merge_pipe, _norm_id, _SEG_SUFFIX
from .utils import require_nonempty_df, format_table
from .mipvu import apply_mipvu_filters
__all__ = ['parse_cql', 'run_cql_query']

def _build_sentence_index(df: pd.DataFrame):
    if 'sentence_id' not in df.columns:
        raise KeyError("DataFrame must contain 'sentence_id'.")
    sent_map: Dict[str, List[int]] = {}
    row2pos: Dict[int, Tuple[str, int]] = {}
    last = object()
    bucket: List[int] = []
    for idx, row in df.iterrows():
        sid = row['sentence_id']
        if sid != last and bucket:
            sent_map[last] = bucket
            bucket = []
        row2pos[idx] = (sid, len(bucket))
        bucket.append(idx)
        last = sid
    if bucket:
        sent_map[last] = bucket
    return (sent_map, row2pos)

def _token_matches(df: pd.DataFrame, idx: int, cond: Dict[str, Any]) -> bool:
    row = df.loc[idx]
    for key, pat in cond.items():
        if key not in row:
            return False
        val = '' if pd.isna(row[key]) else str(row[key])
        if pat is None:
            continue
        if hasattr(pat, 'search'):
            if pat.search(val) is None:
                return False
            continue
        if isinstance(pat, str) and pat.startswith('!'):
            patt = pat[1:]
            try:
                regex = re.compile(patt, re.IGNORECASE)
                if regex.fullmatch(val):
                    return False
            except re.error:
                if val.lower() == patt.lower():
                    return False
            continue
        try:
            regex = re.compile(pat, re.IGNORECASE)
            if not regex.fullmatch(val):
                return False
        except re.error:
            if val.lower() != pat.lower():
                return False
    return True

def _gap_iter(mm: int, MM: Optional[int], pos: int, n: int, has_next: bool):
    if MM is None:
        cap = n - pos if not has_next else max(0, n - pos - 1)
        upper = max(mm, cap)
    else:
        upper = MM
    if mm < 0:
        mm = 0
    if upper < mm:
        return
    for g in range(mm, upper + 1):
        yield g

def _match_sequence(df: pd.DataFrame, row_ids: List[int], pos: int, seq: List[Any], idx: int, n: int):
    if idx >= len(seq):
        return (pos, [])
    elem = seq[idx]
    if isinstance(elem, tuple) and len(elem) == 3 and (elem[0] == 'gap'):
        m_gap, M_gap = (elem[1], elem[2])
        has_next = idx + 1 < len(seq)
        for g in _gap_iter(m_gap, M_gap, pos, n, has_next):
            new_pos = pos + g
            res_pos, res_hits = _match_sequence(df, row_ids, new_pos, seq, idx + 1, n)
            if res_pos is not None:
                return (res_pos, res_hits)
        return (None, None)
    if isinstance(elem, list):
        gp_pos, gp_hits = _match_sequence(df, row_ids, pos, elem, 0, n)
        if gp_pos is None:
            return (None, None)
        tail_pos, tail_hits = _match_sequence(df, row_ids, gp_pos, seq, idx + 1, n)
        if tail_pos is None:
            return (None, None)
        return (tail_pos, gp_hits + tail_hits)
    if pos >= n:
        return (None, None)
    cond = elem.get('attrs', elem) if isinstance(elem, dict) else elem
    if not _token_matches(df, row_ids[pos], cond):
        return (None, None)
    nxt_pos, nxt_hits = _match_sequence(df, row_ids, pos + 1, seq, idx + 1, n)
    if nxt_pos is None:
        return (None, None)
    return (nxt_pos, [pos] + nxt_hits)

def _eval_cql_sequence(df: pd.DataFrame, seq: List[Any], start_positions: Optional[List[int]]=None, anchor_end: bool=False):
    sent_map, _ = _build_sentence_index(df)
    out = []
    for _, row_ids in sent_map.items():
        n = len(row_ids)
        if start_positions is None:
            starts = range(n)
        else:
            starts = sorted(set((p for p in start_positions if 0 <= p < n)))
        for i in starts:
            if not seq:
                matched: List[int] = []
            else:
                _, matched = _match_sequence(df, row_ids, i, seq, 0, n)
                if matched is None:
                    continue
            if anchor_end:
                if matched:
                    if matched[-1] != n - 1:
                        continue
                elif i != n - 1:
                    continue
            out.append((i, matched))
    return out

def parse_cql(cql: str) -> List[Any]:
    s = re.sub('\\s+', ' ', cql.strip())
    n = len(s)
    i = 0

    def skip_ws(j):
        return j

    def parse_values(j):
        j = skip_ws(j)
        if j >= n or s[j] not in ['"', "'"]:
            raise ValueError("Expected quoted value after '='")
        vals = []
        neg = False
        while True:
            quote = s[j]
            j += 1
            start = j
            while j < n and s[j] != quote:
                j += 1
            if j >= n:
                raise ValueError('Unterminated quoted value')
            val = s[start:j]
            j += 1
            if not vals and val.startswith('!'):
                neg = True
                val = val[1:]
            vals.append(val)
            j = skip_ws(j)
            if j < n and s[j] == ',':
                j += 1
                j = skip_ws(j)
                if j >= n or s[j] not in ['"', "'"]:
                    raise ValueError("Expected quoted value after ','")
                continue
            break
        if len(vals) == 1:
            patt = vals[0]
        else:
            patt = '(?:' + '|'.join(vals) + ')'
        if neg:
            patt = '!' + patt
        return (patt, j)

    def parse_token(j):
        j += 1
        j = skip_ws(j)
        if j < n and s[j] == ']':
            j += 1
            return (('gap', 1, 1), j)
        attrs: Dict[str, str] = {}
        while j < n and s[j] != ']':
            start = j
            while j < n and (s[j].isalnum() or s[j] == '_'):
                j += 1
            if j == start:
                raise ValueError('Expected attribute name in token spec')
            name = s[start:j].lower()
            j = skip_ws(j)
            if j >= n or s[j] != '=':
                raise ValueError("Expected '=' after attribute name")
            j += 1
            patt, j = parse_values(j)
            attrs[name] = patt
            j = skip_ws(j)
            if j < n and s[j] in ['&', ',']:
                j += 1
                j = skip_ws(j)
                continue
            if j < n and s[j] != ']':
                raise ValueError("Unexpected content in token; expected ']' or '&'")
        if j >= n or s[j] != ']':
            raise ValueError("Unterminated token '[' ... '']'")
        j += 1
        return ({'attrs': attrs}, j)

    def parse_gap_after_brackets(j):
        j = skip_ws(j)
        if j < n and s[j] == '{':
            j += 1
            j = skip_ws(j)
            start = j
            while j < n and s[j].isdigit():
                j += 1
            if j == start:
                raise ValueError('Gap lower bound must be a number')
            mval = int(s[start:j])
            j = skip_ws(j)
            MM: Optional[int] = None
            if j < n and s[j] == ',':
                j += 1
                j = skip_ws(j)
                start = j
                while j < n and s[j].isdigit():
                    j += 1
                if j > start:
                    MM = int(s[start:j])
                else:
                    MM = None
            else:
                MM = mval
            j = skip_ws(j)
            if j >= n or s[j] != '}':
                raise ValueError("Gap bounds must end with '}'")
            j += 1
            if MM is not None and MM < mval:
                raise ValueError('Gap upper bound must be >= lower bound')
            return (('gap', mval, MM), j)
        return (None, j)

    def parse_group(j):
        j += 1
        depth = 1
        start = j
        while j < n and depth > 0:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0 or j >= n:
            raise ValueError("Unterminated group '(...)'")
        inner = s[start:j]
        j += 1
        elems = parse_cql(inner)
        return (elems, j)
    out: List[Any] = []
    i = skip_ws(0)
    if i < n and s[i] == '^':
        out.append(('anchor', 'start'))
        i += 1
        i = skip_ws(i)
    while i < n:
        ch = s[i]
        if ch == '[':
            elem, i = parse_token(i)
            if isinstance(elem, tuple) and elem[0] == 'gap':
                g, i2 = parse_gap_after_brackets(i)
                if g is not None:
                    elem = g
                    i = i2
            out.append(elem)
            i = skip_ws(i)
        elif ch == '(':
            elem, i = parse_group(i)
            out.append(elem)
            i = skip_ws(i)
        elif ch == '{':
            raise ValueError("A gap quantifier must follow '[]' or a group, not start a pattern")
        elif ch == '$':
            out.append(('anchor', 'end'))
            i += 1
            i = skip_ws(i)
            if i < n:
                raise ValueError("Trailing content after '$' end anchor")
        else:
            if ch in ',;':
                i += 1
                i = skip_ws(i)
                continue
            if ch.isspace():
                i += 1
                i = skip_ws(i)
                continue
            raise ValueError(f"Unexpected character '{ch}' at position {i}")
    return out

def _collect_cql_attrs(seq: List[Any]):
    used: set[str] = set()

    def walk(e):
        if isinstance(e, dict) and 'attrs' in e:
            for k in e['attrs'].keys():
                used.add(str(k).lower())
        elif isinstance(e, list):
            for x in e:
                walk(x)
        elif isinstance(e, tuple) and len(e) == 3 and (e[0] == 'gap'):
            return
    for el in seq:
        walk(el)
    return used

@require_nonempty_df
def run_cql_query(df: pd.DataFrame, cql: str, outdir: str, basename: str, window: Optional[int]=None) -> str:
    if window is None:
        window = config.CONFIG.get('DEFAULT_WINDOW', 15)
    df_f, _, mrw_mask = apply_mipvu_filters(df)
    seq = parse_cql(cql)
    anchor_start = bool(seq and isinstance(seq[0], tuple) and (seq[0][0] == 'anchor') and (seq[0][1] == 'start'))
    anchor_end = bool(seq and isinstance(seq[-1], tuple) and (seq[-1][0] == 'anchor') and (seq[-1][1] == 'end'))
    if anchor_start:
        seq = seq[1:]
    if anchor_end and seq and isinstance(seq[-1], tuple) and (seq[-1][0] == 'anchor'):
        seq = seq[:-1]
    try:
        used_attrs = _collect_cql_attrs(seq)
        lower_cols = {c.lower() for c in df_f.columns}
        unknown = sorted([a for a in used_attrs if a not in lower_cols])
        if unknown:
            config.logger.warning(f'CQL references unknown attributes: {unknown}')
    except Exception:
        config.logger.exception('Error while checking CQL attributes')
    rows = []
    for sid, sent in df_f.groupby('sentence_id', sort=False):
        idxs = sent.index.tolist()
        if not idxs:
            continue
        start_positions: List[int] = []
        if anchor_start:
            start_positions = [0]
        else:
            start_positions = list(range(len(idxs)))
        if seq and isinstance(seq[0], dict) and ('attrs' in seq[0]):
            attrs = seq[0]['attrs']
            mask = pd.Series(True, index=sent.index)
            for k, val in attrs.items():
                col = k.lower()
                if col not in sent.columns:
                    mask &= False
                    continue
                series = sent[col].astype(str)
                if val.startswith('!'):
                    patt = val[1:]
                    if patt:
                        try:
                            regex = re.compile(patt, re.IGNORECASE)
                            mask &= ~series.str.match(regex, na=False)
                        except re.error:
                            mask &= series.str.lower() != patt.lower()
                    else:
                        mask &= True
                else:
                    try:
                        regex = re.compile(val, re.IGNORECASE)
                        mask &= series.str.match(regex, na=False)
                    except re.error:
                        mask &= series.str.lower() == val.lower()
            cand = sent.index[mask].tolist()
            pos_map = {idx: ii for ii, idx in enumerate(idxs)}
            start_positions = [pos_map[c] for c in cand if c in pos_map]
            if not start_positions:
                continue
        spans = _eval_cql_sequence(sent, seq, start_positions, anchor_end=anchor_end)
        for start_rel, _ in spans:
            node_rel = start_rel
            node_idx = idxs[node_rel]
            from .analysis import extract_context
            left = extract_context(df_f, pd.Index([node_idx]), window, 'left', mrw_mask)
            right = extract_context(df_f, pd.Index([node_idx]), window, 'right', mrw_mask)
            rows.append({'sentence_id': sid, 'Left': left[0], 'Node': f"{df_f.at[node_idx, 'word']}_{df_f.at[node_idx, 'pos']}", 'relation_mrw': 'MRW' if mrw_mask.get(node_idx, False) else 'non-MRW', 'Right': right[0], 'query': cql})
    out = pd.DataFrame(rows, columns=['sentence_id', 'Left', 'Node', 'relation_mrw', 'Right', 'query'])
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'{basename}_CQLKwic.csv')
    out.to_csv(path, index=False, encoding='utf-8-sig')
    return path
