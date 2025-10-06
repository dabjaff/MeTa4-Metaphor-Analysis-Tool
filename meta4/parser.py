from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from lxml import etree
from . import config
from .utils import _merge_pipe, _norm_id, infer_genre
from .mipvu import ensure_normalized
__all__ = ['extract_row', 'promote', 'parse_sentence_merge_mwe', 'parse_xml_to_df']

def extract_row(w: etree._Element, fid: str, sid: str, genre: str) -> Dict[str, Any] | None:
    try:
        if config.EXCLUDE_TRUNC and w.xpath(".//tei:seg[@function='trunc']", namespaces=config.NS):
            return None
    except Exception:
        pass
    lemma = (w.get('lemma') or '').strip()
    pos = (w.get('type') or '').strip()
    if config.EXCLUDE_PUNCT and pos == 'PUN':
        return None
    word_text = (w.text or '').strip()
    met = ''
    mtype = ''
    msub = ''
    mflag = ''
    seg_override: str | None = None
    seg_anchor_id = ''
    seg_corresp_id = ''
    try:
        for seg in w.xpath('tei:seg', namespaces=config.NS):
            func = (seg.get('function') or '').strip()
            stype = (seg.get('type') or '').strip()
            ssub = (seg.get('subtype') or '').strip()
            if func == 'mrw':
                met = 'mrw'
                if seg.text:
                    seg_override = seg.text.strip()
                mtype = _merge_pipe(mtype, stype)
                msub = _merge_pipe(msub, ssub)
                if not seg_anchor_id:
                    seg_anchor_id = (seg.get(f"{{{config.NS['xml']}}}id") or '').strip()
                if not seg_corresp_id:
                    seg_corresp_id = (seg.get('corresp') or '').lstrip('#').strip()
            elif func == 'mFlag':
                mflag = 'mFlag'
                if seg.text:
                    seg_override = seg.text.strip()
                mtype = _merge_pipe(mtype, stype)
                msub = _merge_pipe(msub, ssub)
    except Exception:
        pass
    if seg_override:
        word_text = seg_override
    w_xmlid = (w.get(f"{{{config.NS['xml']}}}id") or '').strip()
    w_corresp = (w.get('corresp') or '').lstrip('#').strip()
    xmlid = _norm_id(seg_anchor_id or w_xmlid)
    corresp = _norm_id(seg_corresp_id or w_corresp)
    return {'File_ID': fid, 'Genre': genre, 'Sentence_ID': sid, 'Original_Word': word_text, 'Lemma': lemma, 'POS': pos, 'Metaphor': met, 'Type': mtype, 'Subtype': msub, 'MFlag': mflag, 'xml:id': xmlid, 'corresp': corresp}

def promote(anchor_row: Dict[str, Any], part_row: Dict[str, Any]) -> None:
    anchor_row['Original_Word'] = f"{anchor_row['Original_Word']} {part_row['Original_Word']}".strip()
    if part_row.get('Lemma'):
        anchor_row['Lemma'] = f"{anchor_row['Lemma']} {part_row['Lemma']}".strip()
    if part_row.get('POS'):
        if anchor_row.get('POS'):
            anchor_row['POS'] = f"{anchor_row['POS']}+{part_row['POS']}"
        else:
            anchor_row['POS'] = part_row['POS']
    if part_row.get('Metaphor') == 'mrw':
        anchor_row['Metaphor'] = 'mrw'
    if part_row.get('MFlag') == 'mFlag':
        anchor_row['MFlag'] = 'mFlag'
    anchor_row['Type'] = _merge_pipe(anchor_row.get('Type', ''), part_row.get('Type', ''))
    anchor_row['Subtype'] = _merge_pipe(anchor_row.get('Subtype', ''), part_row.get('Subtype', ''))

def parse_sentence_merge_mwe(s_node: etree._Element, fid: str, genre: str, sid: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    anchor_idx: Dict[str, int] = {}
    try:
        for w in s_node.xpath('.//tei:w', namespaces=config.NS):
            r = extract_row(w, fid, sid, genre)
            if r is None:
                continue
            xmlid = _norm_id(r.get('xml:id') or '')
            cor = _norm_id(r.get('corresp') or '')
            if xmlid and (not cor):
                if xmlid in anchor_idx:
                    promote(rows[anchor_idx[xmlid]], r)
                    rows[anchor_idx[xmlid]]['xml:id'] = xmlid
                    rows[anchor_idx[xmlid]]['corresp'] = ''
                else:
                    rows.append({**r, 'xml:id': xmlid, 'corresp': ''})
                    anchor_idx[xmlid] = len(rows) - 1
                continue
            if cor:
                if cor in anchor_idx:
                    promote(rows[anchor_idx[cor]], r)
                else:
                    placeholder = {'File_ID': fid, 'Genre': genre, 'Sentence_ID': sid, 'Original_Word': r['Original_Word'], 'Lemma': r['Lemma'], 'POS': r['POS'], 'Metaphor': r['Metaphor'], 'Type': r['Type'], 'Subtype': r['Subtype'], 'MFlag': r['MFlag'], 'xml:id': cor, 'corresp': ''}
                    rows.append(placeholder)
                    anchor_idx[cor] = len(rows) - 1
                continue
            rows.append(r)
    except Exception:
        config.logger.exception(f'Failed to parse sentence {sid}')
    return rows

def parse_xml_to_df(xml_path: Path) -> pd.DataFrame:
    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        raise ValueError(f'Failed to parse XML at {xml_path}: {e}')
    root = tree.getroot()
    texts = root.xpath('.//tei:text[@xml:id]', namespaces=config.NS)
    out_rows: List[Dict[str, Any]] = []
    for text in texts:
        fid = text.get(f"{{{config.NS['xml']}}}id") or ''
        genre = infer_genre(fid)
        for s in text.xpath('.//tei:s', namespaces=config.NS):
            if config._is_b1g(fid):
                n_attr = s.get('n')
                try:
                    n_val = int(n_attr) if n_attr is not None else None
                except ValueError:
                    n_val = None
                if n_val is None or not config._b1g_sentence_allowed(n_val):
                    continue
            sid = f"{fid}_s{s.get('n', '')}"
            out_rows.extend(parse_sentence_merge_mwe(s, fid, genre, sid))
        try:
            if not config._is_b1g(fid):
                all_w = text.xpath('.//tei:w', namespaces=config.NS)
                w_in_s = set(text.xpath('.//tei:s//tei:w', namespaces=config.NS))
                outside = [w for w in all_w if w not in w_in_s]
                if outside:
                    count = 0
                    for w in outside:
                        count += 1
                        sid = f'{fid}_nosent{count:04d}'
                        bucket = etree.Element(f"{{{config.NS['tei']}}}s")
                        bucket.append(w)
                        out_rows.extend(parse_sentence_merge_mwe(bucket, fid, genre, sid))
        except Exception:
            config.logger.exception(f'Error while grouping tokens outside of sentence for file {fid}')
    df = pd.DataFrame(out_rows)
    if not df.empty and 'Original_Word' in df.columns:
        df = df[df['Original_Word'].astype(str).str.strip().astype(bool)].copy()
    df = ensure_normalized(df)
    try:
        df.attrs['corpus'] = 'vuamc'
    except Exception:
        pass
    return df
