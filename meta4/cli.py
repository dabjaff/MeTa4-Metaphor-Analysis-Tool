from __future__ import annotations
import os
from typing import List, Optional
from . import config
from .utils import ask, get_menu_choice, show_help, get_timestamp
from .io import load_vuamc_file, load_csv
from .analysis import single_lemma_report_classic, save_concordance_dual, save_collocations_dual, find_lemma_regex_hits, export_pattern_kwic, export_flat_vuamc_csv, export_mrw_list, export_mrw_list_by_pos
from .mipvu import mipvu_counts, ensure_normalized
from .cql import run_cql_query, parse_cql
from .utils import safe_slug
from .parser import parse_xml_to_df
from pathlib import Path
from difflib import get_close_matches
import pandas as pd
__all__ = ['main']
PKG_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = Path(os.getenv('META4_RESULTS_DIR', PKG_ROOT))

def make_outdir(*parts: str) -> Path:
    outdir = RESULTS_BASE.joinpath('results', *parts)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def show_tutorial() -> None:
    print('\nWelcome to MeTa4: Metaphor Analysis Tool!\nThis tool analyzes metaphor usage in text corpora using the MIPVU methodology.\nKey terms:\n  - Lemma: Base form of a word (e.g., \'run\' for \'running\', \'runs\').\n  - MRW: Metaphor-Related Word, identified by MIPVU criteria.\n  - LU: Lexical Unit, the denominator for metaphor density.\n  - CQL: Corpus Query Language for pattern searches (e.g., [lemma="run"] [pos="VB.*"]).\n\nHow to use:\n1. Explore the VUAMC corpus or upload your own XML/CSV/TSV file.\n2. Choose an analysis:\n   • Single Lemma: Analyze a specific lemma (e.g., \'run\').\n   • Batch Lemma: Analyze multiple lemmas.\n   • Collocations: Find words co-occurring with a lemma.\n   • Pattern Search: Use regex or CQL for complex queries.\n3. Results are saved in \'results/\' with timestamps.\n\nPress [R] to return to the previous menu at any time, [H] for help or [T] for this tutorial.\n        ')

def analysis_explorer(df: pd.DataFrame, corpus_name: str='VUAMC') -> None:
    # Ensure positional indices are 0..N-1 for all downstream lookups (KWIC/context)
    df = df.reset_index(drop=True)

    num_mrw, denom = mipvu_counts(df)
    is_user = str(df.attrs.get('corpus', '')).lower() == 'user'
    print(f'\n=== {corpus_name} Overview (MIPVU) ===')
    print(f'  Lexical Units (LUs): {denom}')
    print(f"  Unique lemmas: {df.loc[df['lemma'].astype(str).str.len() > 0, 'lemma'].nunique()}")
    if is_user:
        # For user files: Show breakdown of metaphor_function values
        metaphor_counts = df['metaphor_function'].value_counts()
        if not metaphor_counts.empty:
            total_lus = denom
            print('What are the types of values are in the metaphor column:')
            for i, (value, count) in enumerate(metaphor_counts.items(), 1):
                percentage = (count / total_lus * 100) if total_lus > 0 else 0
                print(f'  {i}. {value} ({count}) {percentage:.2f}%')
        else:
            print('  No values found in metaphor_function column.')
    else:
        # For VUAMC: Standard MRW summary
        print(f'  Metaphor-Related Words (MRWs): {num_mrw}')
        print(f'  % MRW: {(num_mrw / denom * 100 if denom else 0):.2f}%')
        print(f"  MRW per {config.CONFIG['PER_1000']} LUs: {(num_mrw / denom * config.CONFIG['PER_1000'] if denom else 0):.2f}\n")
    if is_user:
        print()  # Extra newline for spacing
    while True:
        print(f'{corpus_name} Analysis Menu')
        print('  [1] Single Lemma Analysis\n  [2] Batch Lemma Analysis')
        print('  [3] Collocation Analysis, Lemma\n  [4] Pattern Search (regex/CQL), Word form (pre-normalization)')
        print('  [5] Print (export CSVs)')
        print('  [H] Help (query power)\n  [T] Tutorial (beginner guide)')
        print('  [R] Return to Main Menu\n  [Q] Quit')
        mode, val = get_menu_choice(valid={'1', '2', '3', '4', '5'}, context='analysis')
        if mode == 'help':
            print('\n' + config.HELP['patterns'] + '\n')
            continue
        if mode == 'tutorial':
            show_tutorial()
            continue
        if mode == 'return':
            return
        if mode == 'quit':
            config.logger.info('Exiting MeTa4')
            return
        if mode != 'select':
            config.logger.warning('Invalid choice.')
            continue
        c = val
        if c == '1':
            show_help('single')
            resp = ask("Enter a lemma (e.g., 'run'): ", context='single')
            if resp is None:
                continue
            if resp in ('__HOTKEY_HELP__', '__HOTKEY_TUTORIAL__'):
                continue
            lemma = resp.strip().lower()
            try:
                text = single_lemma_report_classic(df, lemma, df)
                print(text)
                if 'No lemma' in text or 'did you mean' in text:
                    continue
                stamp = get_timestamp()
                outdir = make_outdir(f'{corpus_name}_Lemm_{safe_slug(lemma)}_{stamp}')
                with open(outdir / f'analysis_{safe_slug(lemma)}.txt', 'w', encoding='utf-8') as f:
                    f.write(text)
                save_concordance_dual(df, lemma, str(outdir), stamp=stamp)
                print(f'Saved lemma analysis to: {outdir}')
            except ValueError as e:
                config.logger.warning(f'Analysis error: {str(e)}')
        elif c == '2':
            show_help('batch')
            resp = ask('Enter lemmas (comma-separated or a filename): ', context='batch')
            if resp is None:
                continue
            if resp in ('__HOTKEY_HELP__', '__HOTKEY_TUTORIAL__'):
                continue
            raw = resp.strip()
            items: List[str] = []
            if os.path.isfile(raw):
                try:
                    with open(raw, encoding='utf-8') as f:
                        items = [l.strip() for l in f if l.strip()]
                except Exception as e:
                    config.logger.warning(f"Failed to read lemma file '{raw}': {e}")
                    continue
            else:
                items = [l.strip() for l in raw.split(',') if l.strip()]
            if not items:
                config.logger.warning('No lemmas provided.')
                continue
            stamp = get_timestamp()
            outdir = make_outdir(f'{corpus_name}_Batch_{stamp}')
            for lemma in items:
                try:
                    text = single_lemma_report_classic(df, lemma.lower(), df)
                    with open(outdir / f'analysis_{safe_slug(lemma)}.txt', 'w', encoding='utf-8') as f:
                        f.write(text)
                    save_concordance_dual(df, lemma.lower(), str(outdir), stamp=stamp)
                except ValueError as e:
                    config.logger.warning(f"Batch analysis error for '{lemma}': {e}")
            print(f'Saved batch analyses to: {outdir}')
        elif c == '3':
            show_help('collocations')
            resp = ask('Enter a lemma for collocation analysis: ', context='collocations')
            if resp is None:
                continue
            if resp in ('__HOTKEY_HELP__', '__HOTKEY_TUTORIAL__'):
                continue
            lemma = resp.strip().lower()
            if not lemma:
                config.logger.warning('Empty lemma provided.')
                continue
            stamp = get_timestamp()
            outdir = make_outdir(f'{corpus_name}_Colloc_{safe_slug(lemma)}_{stamp}')
            save_collocations_dual(df, lemma, str(outdir), stamp=stamp)
            print(f'Saved collocation analysis to: {outdir}')
        elif c == '4':
            show_help('patterns')
            resp = ask('Enter a regex (lemma) or CQL pattern: ', context='patterns')
            if resp is None:
                continue
            if resp in ('__HOTKEY_HELP__', '__HOTKEY_TUTORIAL__'):
                continue
            q = resp.strip()
            if not q:
                config.logger.warning('Empty pattern provided.')
                continue
            stamp = get_timestamp()
            outdir = make_outdir(f'{corpus_name}_Patt_{stamp}')
            try:
                if any((q.lstrip().startswith(ch) for ch in ['[', '(', '^', '$'])):
                    basename = safe_slug(q)
                    path = run_cql_query(df, q, str(outdir), basename)
                    print(f'Saved CQL KWIC to: {path}')
                else:
                    hits = find_lemma_regex_hits(df, q)
                    spans = []
                    for sid, group in df.groupby('sentence_id'):
                        indices = [idx for idx in group.index if idx in hits]
                        if indices:
                            spans.append((sid, indices))
                    if not spans:
                        print(f"No matches found for pattern '{q}'.")
                        continue
                    path = outdir / f'{safe_slug(q)}_Kwic.csv'
                    export_pattern_kwic(df, spans, str(path), q, raw_mode=False)
                    print(f'Saved regex KWIC to: {path}')
            except Exception as e:
                config.logger.warning(f'Pattern search error: {e}')
        elif c == '5':
            while True:
                print('\nExport Menu')
                print('  [1] Full flat CSV (all tokens)\n  [2] MRW list\n  [3] MRW list by POS')
                print('  [H] Help\n  [R] Return\n  [Q] Quit')
                mode2, val2 = get_menu_choice(valid={'1', '2', '3'}, context='print')
                if mode2 == 'help':
                    show_help('upload')
                    continue
                if mode2 == 'return':
                    break
                if mode2 == 'quit':
                    config.logger.info('Exiting MeTa4')
                    return
                if mode2 != 'select':
                    config.logger.warning('Invalid choice.')
                    continue
                cc = val2
                stamp = get_timestamp()
                outdir = make_outdir(f'{corpus_name}_Print_{stamp}')
                try:
                    if cc == '1':
                        path = outdir / f'{corpus_name}_Full.csv'
                        export_flat_vuamc_csv(df, str(path))
                        print(f'Saved full flat CSV to: {path}')
                    elif cc == '2':
                        path = outdir / f'{corpus_name}_MRW_List.csv'
                        export_mrw_list(df, str(path))
                        print(f'Saved MRW list to: {path}')
                    elif cc == '3':
                        resp_pos = ask("Enter POS tag (e.g., 'NN'): ", context='print')
                        if resp_pos is None:
                            continue
                        if resp_pos in ('__HOTKEY_HELP__', '__HOTKEY_TUTORIAL__'):
                            continue
                        pos_tag = resp_pos.strip()
                        if not pos_tag:
                            config.logger.warning('Empty POS provided.')
                            continue
                        path = outdir / f'{corpus_name}_MRW_POS_{pos_tag}.csv'
                        export_mrw_list_by_pos(df, pos_tag, str(path))
                        print(f'Saved MRW-by-POS list to: {path}')
                except ValueError as e:
                    config.logger.warning(f'Print export error: {e}')

def vuamc_explorer() -> None:
    try:
        vuamc = load_vuamc_file()
        if vuamc is None:
            return
        # Set corpus attr for VUAMC
        vuamc.attrs['corpus'] = 'vuamc'
        print('\n\n' + config.HELP['vuamc'].strip() + '\n\n', end='')
        analysis_explorer(ensure_normalized(vuamc), 'VUAMC')
    except FileNotFoundError as e:
        config.logger.warning(f'VUAMC loading error: {str(e)}')
        print('Failed to load VUAMC data. Please ensure the file exists and is accessible.')
        return

def genre_level_explorer() -> None:
    try:
        vuamc = load_vuamc_file()
    except FileNotFoundError as e:
        config.logger.warning(f'VUAMC loading error: {str(e)}')
        print('Failed to load VUAMC data. Please ensure the file exists and is accessible.')
        return
    if vuamc is None:
        return
    # Ensure corpus attr is set for genre slices too
    if 'corpus' not in vuamc.attrs:
        vuamc.attrs['corpus'] = 'vuamc'
    if 'genre' not in vuamc.columns:
        config.logger.warning("No 'genre' column found. Falling back to whole corpus.")
        analysis_explorer(ensure_normalized(vuamc), 'VUAMC')
        return
    genres = sorted(set((str(g).strip() for g in vuamc['genre'].dropna().astype(str) if str(g).strip())))
    if not genres:
        print('No genres available in the loaded data.')
        return
    print('\nAvailable genres:')
    for i, g in enumerate(genres, 1):
        print(f'  [{i}] {g}')
    print('  [H] Help')
    print('  [T] Tutorial')
    print('  [R] Return')
    print('  [Q] Quit')
    print()
    chosen: Optional[str] = None
    while True:
        valid = {str(i) for i in range(1, len(genres) + 1)}
        mode, val = get_menu_choice(valid=valid, context='genre', prompt='Pick a genre (number or name): ')
        if mode == 'help':
            show_help('vuamc')
            continue
        if mode == 'tutorial':
            show_tutorial()
            continue
        if mode == 'return':
            return
        if mode == 'quit':
            config.logger.info('Exiting MeTa4')
            return
        if mode != 'select':
            choice = str(val).strip().lower()
            if choice:
                matches = [g for g in genres if g.lower() == choice]
                if matches:
                    chosen = matches[0]
                    break
            config.logger.warning('Invalid choice.')
            continue
        idx = int(val)
        chosen = genres[idx - 1]
        break
    if not chosen:
        print('Unrecognized selection. Returning.')
        return
    # Critical: reset index on the genre slice so positions are 0..N-1
    df_g = vuamc[vuamc['genre'].astype(str).str.lower() == chosen.lower()].copy().reset_index(drop=True)
    # Preserve corpus attr on slice
    df_g.attrs['corpus'] = 'vuamc'
    print(f'\n\nVUAMC Explorer — Genre: {chosen}\n')
    analysis_explorer(ensure_normalized(df_g), f'VUAMC:{chosen}')

def user_file_explorer() -> None:
    def _normalize_user_headers(df):
        cols = (
            pd.Index(df.columns)
              .map(lambda c: str(c).strip())
              .str.replace(r'\s+', '_', regex=True)
              .str.lower()
        )
        df.columns = cols
        if 'lemma' not in df.columns and 'word' in df.columns:
            try:
                df['lemma'] = df['word'].astype(str).str.lower()
            except Exception:
                df['lemma'] = df['word']
        if 'pos' not in df.columns:
            df['pos'] = 'X'
        if 'genre' not in df.columns:
            df['genre'] = 'User'
        if 'sentence_id' not in df.columns:
            df['sentence_id'] = range(1, len(df) + 1)
        # Add defaults for MIPVU-specific columns to prevent KeyErrors in analysis
        if 'type' not in df.columns:
            df['type'] = ''
        if 'subtype' not in df.columns:
            df['subtype'] = ''
        if 'mflag' not in df.columns:
            df['mflag'] = ''
        return df

    # Updated: Include Excel files in scan
    files = [f for f in os.listdir('.') if f.lower().endswith(('.xml', '.zip', '.csv', '.tsv', '.xlsx', '.xls'))]
    if not files:
        config.logger.warning('No compatible files (.xml, .zip, .csv, .tsv, .xlsx, .xls) found.')
        print('No compatible files found in the current directory.')
        return
    try:
        show_help('upload')
    except Exception:
        pass
    while True:
        print('\nUser File Analysis')
        for i, f in enumerate(files, 1):
            print(f'  [{i}] {f}')
        print('  [H] Help')
        print('  [I] Instructions')
        print('  [E] Example CSV')
        print('  [R] Return to Main Menu')
        print('  [Q] Quit')
        valid = {str(i) for i in range(1, len(files) + 1)}
        mode, val = get_menu_choice(valid=valid, context='analysis')
        if mode == 'help':
            try:
                show_help('upload')
            except Exception:
                print('Help unavailable.')
            continue
        if mode == 'tutorial':
            show_tutorial()
            continue
        if mode == 'return':
            return
        if mode == 'quit':
            config.logger.info('Exiting MeTa4')
            return
        if mode == 'invalid' and str(val).lower() == 'i':
            try:
                show_help('upload')
            except Exception:
                print('Instructions unavailable.')
            continue
        if mode == 'invalid' and str(val).lower() == 'e':
            try:
                _show_upload_example_csv()
            except Exception:
                print('Example CSV unavailable.')
            continue
        if mode != 'select':
            config.logger.warning('Invalid choice.')
            continue
        sel = val
        if sel.isdigit() and 1 <= int(sel) <= len(files):
            chosen = files[int(sel) - 1]
            path = Path(chosen)  # Use Path for suffix access
            try:
                ext = str(path.suffix).lower()
                if chosen.lower().endswith(('.xml', '.zip')):
                    df = load_vuamc_file() if chosen.lower().startswith('vuamc') else parse_xml_to_df(path)
                elif ext in {'.csv', '.tsv'}:
                    df = load_csv(chosen)
                # New: Excel handler
                elif ext in {'.xlsx', '.xls'}:
                    try:
                        df = pd.read_excel(path, sheet_name=0, engine='openpyxl' if ext == '.xlsx' else 'xlrd')
                    except ImportError as ie:
                        raise ImportError(f'Excel support requires openpyxl (xlsx) or xlrd (xls). Install with: pip install openpyxl xlrd==1.2.0. Error: {ie}')
                else:
                    raise ValueError(f'Unsupported file type: {ext}')
                df = _normalize_user_headers(df)
                # Set corpus attr for user file (enables flexible MIPVU logic)
                df.attrs['corpus'] = 'user'
                col_map = {c.strip().lower(): c for c in df.columns}
                missing = [col for col in config.CONFIG.get('USER_REQUIRED', ['lemma', 'word', 'metaphor_function', 'pos']) if col not in col_map]
                if missing:
                    raise ValueError("Missing required columns: " + ", ".join(missing) + ". Required (User Upload): word, metaphor_function. Headers are case-insensitive.")
                analysis_explorer(ensure_normalized(df), os.path.basename(chosen))
            except Exception as e:
                config.logger.exception(f'File loading error: {str(e)}')
                print(f'Error loading file: {str(e)}')
        else:
            config.logger.warning('Invalid selection.')
            print('Invalid selection. Choose a listed number, or press H/I/E/R/Q.')

def main() -> None:
    print('\nWelcome to MeTa4: Metaphor Analysis Tool')
    print('Analyze metaphor usage in text corpora using MIPVU methodology.')
    while True:
        print('\nMain Menu\nWhat to do?\n  [1] Explore the VUAMC\n  [2] Genre-Level Analysis\n  [3] Upload Your File\n  [T] Tutorial\n  [Q] Quit')
        mode, val = get_menu_choice(valid={'1', '2', '3'}, context='main')
        if mode == 'help':
            print('\n' + config.HELP['patterns'] + '\n')
            continue
        if mode == 'tutorial':
            show_tutorial()
            continue
        if mode == 'return':
            return
        if mode == 'quit':
            config.logger.info('Exiting MeTa4')
            print('Goodbye!')
            return
        if mode != 'select':
            config.logger.warning('Invalid choice.')
            continue
        choice = val
        if choice == '1':
            vuamc_explorer()
        elif choice == '2':
            genre_level_explorer()
        elif choice == '3':
            user_file_explorer()
        else:
            config.logger.warning('Invalid choice.')

def _show_upload_example_csv() -> None:
    try:
        import pandas as _pd
        from datetime import datetime as _dt
        _cols = ['file_id', 'sentence_id', 'word', 'lemma', 'pos', 'type', 'subtype', 'mflag', 'genre']
        _rows = [{'file_id': 'a1e_0001', 'sentence_id': 1, 'word': 'Time', 'lemma': 'time', 'pos': 'NN', 'type': '', 'subtype': '', 'mflag': '', 'genre': 'News'}, {'file_id': 'a1e_0001', 'sentence_id': 1, 'word': 'flies', 'lemma': 'fly', 'pos': 'VBZ', 'type': 'MRW', 'subtype': 'direct', 'mflag': 'mrw', 'genre': 'News'}, {'file_id': 'a1e_0001', 'sentence_id': 1, 'word': 'quickly', 'lemma': 'quickly', 'pos': 'RB', 'type': '', 'subtype': '', 'mflag': '', 'genre': 'News'}]
        _outdir = make_outdir(_dt.now().strftime('%Y%m%d_%H%M%S'))
        _path = _outdir / 'example_upload.csv'
        _pd.DataFrame(_rows, columns=_cols).to_csv(str(_path), index=False, encoding='utf-8-sig')
        print(f'Example CSV saved to: {_path}')
        print('You can open it and adjust your own data to match these headers.')
    except Exception as _e:
        config.logger.warning(f'Failed to create example CSV: {_e}')
        print('Could not create the example CSV. Please ensure you have write permissions.')
