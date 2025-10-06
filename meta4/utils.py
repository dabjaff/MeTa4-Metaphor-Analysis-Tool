from __future__ import annotations
import re
import unicodedata
from datetime import datetime
from typing import List, Iterable, Callable, Optional, Any, Tuple
import pandas as pd
from . import config
__all__ = ['ask', 'get_menu_choice', 'show_help', 'get_timestamp', 'safe_slug', 'format_table', 'require_nonempty_df', '_merge_pipe', '_norm_id', 'infer_genre', '_SEG_SUFFIX', '_cache_put']

def ask(prompt: str, context: str | None=None) -> Optional[str]:
    try:
        s = input(prompt).strip()
    except EOFError:
        raise SystemExit(0)
    low = s.lower()
    if len(low) == 1 and low in {'r', 'h', 't', 'q'}:
        if low == 'q':
            raise SystemExit(0)
        if low == 'r':
            return None
        if low == 'h':
            try:
                if context:
                    show_help(context)
            except Exception as e:
                config.logger.debug(f'Help failed: {e}')
            return '__HOTKEY_HELP__'
        if low == 't':
            try:
                from .cli import show_tutorial
                show_tutorial()
            except Exception as e:
                config.logger.debug(f'Tutorial failed: {e}')
            return '__HOTKEY_TUTORIAL__'
    return s

def get_menu_choice(valid: Iterable[str], context: str, prompt: str='Choice: ') -> Tuple[str, Optional[str]]:
    s = ask(prompt, context=context)
    if s is None:
        return ('return', None)
    if s == '__HOTKEY_HELP__':
        return ('help', None)
    if s == '__HOTKEY_TUTORIAL__':
        return ('tutorial', None)
    s = str(s).strip().lower()
    if s == 'r':
        return ('return', None)
    if s == 'q':
        return ('quit', None)
    if s in valid:
        return ('select', s)
    return ('invalid', s)

def show_help(key: str) -> None:
    try:
        text = config.HELP.get(key, config.HELP.get('tutorial', ''))
        print('\n' + text + '\n')
    except Exception as e:
        config.logger.debug(f'Failed to show help for {key}: {e}')

def get_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def safe_slug(s: str, allow_unicode: bool=False) -> str:
    s = str(s or '').strip()
    if allow_unicode:
        slug = unicodedata.normalize('NFKC', s)
    else:
        slug = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    slug = slug.lower()
    slug = re.sub('[\\s\\-]+', '_', slug)
    slug = re.sub('[^\\w\\._]+', '', slug)
    slug = slug.strip('_')
    return slug or 'untitled'

def format_table(headers: List[str], rows: List[Iterable[Any]]) -> str:
    str_rows = [[str(item) for item in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
            else:
                widths.append(len(cell))
    line = ''
    sep = ''
    for i, h in enumerate(headers):
        w = widths[i]
        line += f'{h:<{w}}  '
        sep += f"{'-' * w}  "
    out_lines = [line.rstrip(), sep.rstrip()]
    for row in str_rows:
        rline = ''
        for i, cell in enumerate(row):
            w = widths[i]
            rline += f'{cell:<{w}}  '
        out_lines.append(rline.rstrip())
    return '\n'.join(out_lines) + '\n'

def require_nonempty_df(func: Callable[[pd.DataFrame, Any], Any]) -> Callable[[pd.DataFrame, Any], Any]:

    def wrapper(df: pd.DataFrame, *args, **kwargs):
        if df is None or df.empty:
            print('No data available in this corpus or genre. Please select a different option or check the data source.')
            if func.__name__ == 'mipvu_counts':
                return (0, 0)
            return None
        return func(df, *args, **kwargs)
    wrapper.__name__ = getattr(func, '__name__', 'wrapped_func')
    return wrapper

def _merge_pipe(a: str, b: str) -> str:
    items: List[str] = []
    seen: set[str] = set()
    for part in (a, b):
        for tok in str(part).split('|'):
            tok = tok.strip()
            if tok and tok.lower() not in seen:
                seen.add(tok.lower())
                items.append(tok)
    return '|'.join(items)
_SEG_SUFFIX = re.compile('^(?P<root>.+?)(?:s\\d+)?$')

def _norm_id(x: str) -> str:
    if not x:
        return ''
    m = _SEG_SUFFIX.match(str(x))
    return m.group('root') if m else str(x)

def _cache_put(key: Any, val: Any) -> None:
    try:
        config._cache_put(key, val)
    except Exception as e:
        config.logger.debug(f'Cache put failed: {e}')

def infer_genre(fid: str) -> str:
    try:
        key = (fid or '')[:3].lower()
        return config.prefix_to_genre.get(key, 'Unknown')
    except Exception:
        return 'Unknown'
