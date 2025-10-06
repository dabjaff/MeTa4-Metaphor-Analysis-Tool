from __future__ import annotations
import os
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import pandas as pd
from functools import lru_cache
from . import config
from .parser import parse_xml_to_df
from .utils import ask
__all__ = ['load_csv', 'load_vuamc_file']

def load_csv(file: str, encodings: Optional[Tuple[str, ...]]=None) -> pd.DataFrame:
    if encodings is None:
        encodings = ('utf-8', 'latin1', 'iso-8859-1')
    for enc in encodings:
        try:
            df = pd.read_csv(file, sep=None, engine='python', encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            config.logger.warning(f"Failed to read '{file}' with encoding {enc}: {str(e)}")
    raise FileNotFoundError(f"Failed to read '{file}' after trying encodings: {', '.join(encodings)}")

def _contains_vuamc_xml(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return any((Path(n).name.casefold() == 'vuamc.xml' for n in zf.namelist()))
    except Exception:
        return False

def _load_xml_or_zip(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Path not found: {p}')
    if p.suffix.lower() == '.xml':
        return parse_xml_to_df(p)
    if p.suffix.lower() == '.zip':
        with zipfile.ZipFile(p) as z:
            xml_members = [m for m in z.namelist() if m.lower().endswith('.xml')]
            if not xml_members:
                raise ValueError('Zip file contains no .xml.')
            xml_members.sort(key=len)
            with z.open(xml_members[0]) as fh:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as t:
                    t.write(fh.read())
                    tmp = t.name
        try:
            return parse_xml_to_df(Path(tmp))
        finally:
            try:
                os.remove(tmp)
            except Exception as e:
                config.logger.debug(f'Failed to remove temp file {tmp}: {e}')
    raise ValueError(f'Unsupported file type for XML loader: {p.suffix}')

def _is_valid_vuamc_path(path: str) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    name = p.name.casefold()
    if name == 'vuamc.xml' and p.suffix.lower() == '.xml':
        return True
    if p.suffix.lower() == '.zip':
        return _contains_vuamc_xml(p)
    return False

def _discover_vuamc(search_roots=None) -> Optional[str]:
    if search_roots is None:
        search_roots = [os.getcwd(), str(Path(__file__).resolve().parent), '/mnt/data']
    xml_candidates = []
    zip_candidates = []
    seen = set()
    for root in search_roots:
        r = Path(root)
        if not r.exists() or not r.is_dir():
            continue
        try:
            rp = r.resolve()
            if rp in seen:
                continue
            seen.add(rp)
        except Exception:
            pass
        for path in r.rglob('*'):
            if not path.is_file():
                continue
            try:
                if path.name.casefold() == 'vuamc.xml' and path.suffix.lower() == '.xml':
                    xml_candidates.append(path)
                elif path.suffix.lower() == '.zip' and _contains_vuamc_xml(path):
                    zip_candidates.append(path)
            except Exception:
                continue
    pool = xml_candidates or zip_candidates
    if not pool:
        return None

    def _rank(p: Path):
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        return (p.suffix.lower() == '.xml', size)
    try:
        best = max(pool, key=_rank)
    except Exception:
        best = pool[0]
    return str(best)

@lru_cache(maxsize=1)
def _cached_vuamc_df(resolved_path: str, mtime: float):
    return _load_xml_or_zip(resolved_path)

def load_vuamc_file() -> pd.DataFrame:
    default = config.CONFIG.get('VUAMC_FILE', '').strip()
    if default and _is_valid_vuamc_path(default):
        p = Path(default)
        mtime = os.path.getmtime(p)
        return _cached_vuamc_df(str(p), mtime).copy()
    env_path = os.getenv('METAFY_VUAMC', '').strip()
    if env_path and _is_valid_vuamc_path(env_path):
        p = Path(env_path)
        mtime = os.path.getmtime(p)
        return _cached_vuamc_df(str(p), mtime).copy()
    auto = _discover_vuamc()
    if auto:
        p = Path(auto)
        mtime = os.path.getmtime(p)
        return _cached_vuamc_df(str(p), mtime).copy()
    config.logger.warning('Default VUAMC file not found. Enter path to VUAMC .xml or .zip (containing VUAMC.xml):')
    for _ in range(5):
        resp = ask('Path> ', context='path')
        if resp is None:
            return None
        if resp in ('__HOTKEY_HELP__', '__HOTKEY_TUTORIAL__'):
            continue
        user = resp.strip()
        if user.lower() == 'q':
            raise SystemExit(0)
        if _is_valid_vuamc_path(user):
            try:
                config.set_config('VUAMC_FILE', user)
            except Exception:
                config.logger.exception('Failed to update VUAMC_FILE configuration; using provided path directly')
            p = Path(user)
            mtime = os.path.getmtime(p)
            return _cached_vuamc_df(str(p), mtime).copy()
        config.logger.debug('Not valid. Expecting VUAMC.xml or a .zip that contains it. Try again.')
    raise SystemExit(1)
