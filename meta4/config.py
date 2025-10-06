import os
import logging
from collections import OrderedDict
__all__ = ['CONFIG', 'NS', 'prefix_to_genre', 'EXCLUDE_PUNCT', 'EXCLUDE_TRUNC', 'HELP', 'logger', '_CONTEXT_CACHE', '_CONTEXT_CACHE_MAX', '_cache_put', '_is_b1g', '_b1g_sentence_allowed']
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
_CONTEXT_CACHE: 'OrderedDict[tuple[int, int, str, int, int], str]' = OrderedDict()
_CONTEXT_CACHE_MAX = 200000

def _cache_put(key, val):
    _CONTEXT_CACHE[key] = val
    try:
        _CONTEXT_CACHE.move_to_end(key)
    except Exception as e:
        logger.debug(f'Cache move failed: {e}')
    if len(_CONTEXT_CACHE) > _CONTEXT_CACHE_MAX:
        try:
            _CONTEXT_CACHE.popitem(last=False)
        except Exception:
            _CONTEXT_CACHE.clear()
from types import MappingProxyType
_DEFAULT_CONFIG: dict[str, object] = {'VUAMC_FILE': os.getenv('METAFY_VUAMC', '/mnt/data/VUAMC.xml'), 'DEFAULT_WINDOW': 15, 'COLLOCATION_WINDOW': 5, 'OMEGA_LIMIT': 3, 'PER_1000': 1000, 'USER_REQUIRED': ['lemma', 'word', 'metaphor_function', 'pos']}
CONFIG = MappingProxyType(_DEFAULT_CONFIG)

def set_config(key: str, value: object) -> None:
    if key not in _DEFAULT_CONFIG:
        raise KeyError(f'Unknown configuration key: {key}')
    _DEFAULT_CONFIG[key] = value
NS = {'tei': 'http://www.tei-c.org/ns/1.0', 'xml': 'http://www.w3.org/XML/1998/namespace'}
EXCLUDE_PUNCT = True
EXCLUDE_TRUNC = True
prefix_to_genre = {'a1e': 'News', 'a1f': 'News', 'a1g': 'News', 'a1h': 'News', 'a1j': 'News', 'a1k': 'News', 'a1l': 'News', 'a1m': 'News', 'a1n': 'News', 'a1p': 'News', 'a1u': 'News', 'a1x': 'News', 'a2d': 'News', 'a31': 'News', 'a36': 'News', 'a38': 'News', 'a39': 'News', 'a3c': 'News', 'a3e': 'News', 'a3k': 'News', 'a3m': 'News', 'a3p': 'News', 'a4d': 'News', 'a5e': 'News', 'a7s': 'News', 'a7t': 'News', 'a7w': 'News', 'a7y': 'News', 'a80': 'News', 'a8m': 'News', 'a8n': 'News', 'a8r': 'News', 'a8u': 'News', 'a98': 'News', 'a9j': 'News', 'aa3': 'News', 'ahb': 'News', 'ahc': 'News', 'ahd': 'News', 'ahe': 'News', 'ahf': 'News', 'ahl': 'News', 'ajf': 'News', 'al0': 'News', 'al2': 'News', 'al5': 'News', 'kb7': 'Conversation', 'kbc': 'Conversation', 'kbd': 'Conversation', 'kbh': 'Conversation', 'kbj': 'Conversation', 'kbp': 'Conversation', 'kbw': 'Conversation', 'kcc': 'Conversation', 'kcf': 'Conversation', 'kcu': 'Conversation', 'kcv': 'Conversation', 'a6u': 'Academic', 'acj': 'Academic', 'alp': 'Academic', 'amm': 'Academic', 'as6': 'Academic', 'b17': 'Academic', 'b1g': 'Academic', 'clp': 'Academic', 'clw': 'Academic', 'crs': 'Academic', 'cty': 'Academic', 'ea7': 'Academic', 'ecv': 'Academic', 'ew1': 'Academic', 'fef': 'Academic', 'ab9': 'Fiction', 'ac2': 'Fiction', 'bmw': 'Fiction', 'bpa': 'Fiction', 'c8t': 'Fiction', 'cb5': 'Fiction', 'ccw': 'Fiction', 'cdb': 'Fiction', 'faj': 'Fiction', 'fet': 'Fiction', 'fpb': 'Fiction', 'g0l': 'Fiction'}
HELP = {'main': 'Welcome to MeTa4 (MIPVU)\nAnalyze metaphor use in corpora. Choose a mode or press [T] for a 1‑minute tour.\nOutputs go to results/<timestamped>/.\n', 'vuamc': 'VUAMC Explorer\n- Loads the VU Amsterdam Metaphor Corpus (XML/ZIP) from the script’s directory.\n- Removes ~17K unannotated words from B1G.\n- Merges multi‑word metaphors.\n- Excludes 125 mistaken “of” annotations in the News register from MRW counts.\nTip: Use Genre‑Level Analysis to focus on a subset.\n', 'genre': 'Genre‑Level Analysis\n- Pick one VUAMC genre, then run the same tools (lemma, batch, colloc, patterns).\n- Counts follow MIPVU: MRW after filters; LU excludes punctuation.\n', 'upload': "Upload Instructions (User Files)\nSupported formats:\n  • CSV or TSV with headers\n  • VUAMC XML or a ZIP containing VUAMC‑style XML files\nRequired columns for CSV/TSV (case‑insensitive headers):\n  • lemma\n  • word\n  • metaphor_function\n  • pos\nTips:\n  • Put your file in the same folder where you run this script.\n  • Choose [User File Analysis] from the main menu, select your file by number.\n  • If you see 'Missing required columns', rename your headers to exactly the required names.\n  • Use [E] Example CSV to see the expected layout.\n  • For XML/ZIP, the loader will parse tokens and apply MIPVU‑style filters automatically.\n", 'analysis': 'Analysis Menu\n[1] Single Lemma — counts + KWIC (MRW / non‑MRW CSVs)\n[2] Batch Lemmas — run #1 for many lemmas\n[3] Collocations — L/R distance counts within a window\n[4] Pattern Search — regex or CQL, KWIC export\n[H] Query Help — regex/CQL cheatsheet\n[T] Tutorial — 1‑minute tips\n', 'single': 'Single Lemma\n- Raw hits, LU‑based rates, MRW vs non‑MRW split.\n- Exports two KWIC CSVs (MRW / non‑MRW) with Ω marking.\n', 'batch': 'Batch Lemmas\n- Enter comma‑separated lemmas or a file path (one per line).\n- Produces per‑lemma summaries and KWICs.\n', 'collocations': 'Collocations\n- Counts left/right collocates by distance (±1..window).\n- Without sentence_id: distances use row order per file/group.\n', 'patterns': 'Pattern Search — Quick Help\nRegex (lemma): run*  |  ^re-.*  |  (make|do)\nCQL: [lemma="time"] [pos="VB.*"]\n     ^ [pos="DT"] [lemma="city"] $\n     [lemma="run"] {1,3} [pos="NN.*"]\n     [pos="NN.*" & !lemma="time"]\nAttributes: lemma, pos, word (+ type, subtype, mflag, genre if present)\n', 'tutorial': 'Quick Tips\n- LUs exclude punctuation; MRWs follow MIPVU filters.\n- Uploads: sentence_id not required; contexts use row proximity.\n- VUAMC XML may drop punct earlier; contexts can look tighter than CSV.\n- Outputs in results/<timestamped>/.\n'}
_B1G_ALLOWED_SINGLE = {1012, 1299, 1401}
_B1G_ALLOWED_RANGES = [(738, 765), (1485, 1584)]

def _is_b1g(fid: str) -> bool:
    return (fid or '').lower().startswith('b1g')

def _b1g_sentence_allowed(n_val: int) -> bool:
    if n_val in _B1G_ALLOWED_SINGLE:
        return True
    for lo, hi in _B1G_ALLOWED_RANGES:
        if lo <= n_val <= hi:
            return True
    return False
