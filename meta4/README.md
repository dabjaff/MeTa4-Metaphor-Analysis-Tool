# MeTa4: Metaphor Analysis Tool — Methodology & Package Overview

**MeTa4** is a corpus analysis tool for identifying and exploring metaphor usage following the **MIPVU** procedure. It supports the **VU Amsterdam Metaphor Corpus (VUAMC)** and user-provided CSV/TSV/XML/ZIP datasets. It reports MRW counts and densities, exports KWIC concordances, executes regex/CQL searches, and produces tidy CSV outputs.

> For installation, launching, platform notes, and troubleshooting, see **LAUNCHER.md** (authoritative startup guide).

---

## 1) Corpus Logic (MIPVU‑aware processing)

### 1.1 Lexical Units (LU) — denominator
- Punctuation POS are excluded from the LU denominator in all MRW rate calculations.

### 1.2 Sentence handling (VUAMC specifics)
- **B1G subset:** only whitelisted sentence numbers are loaded (singles such as `1012, 1299, 1401`; ranges such as `738–765`, `1485–1584`). Non‑listed B1G sentences are ignored.
- **Tokens outside `<s>` (non‑B1G):** wrapped into synthetic sentence IDs like `nosent####` to avoid data loss.

### 1.3 MRW mask
- Rows marked as DFMA/DFMA_PUNCT are removed from analysis.
- Excludes `type ∈ {lex, morph, phrase}` and rows that are pure `mflag` annotations.

### 1.4 Truncation
- Segments under `seg@function="trunc"` are skipped.

### 1.5 Known correction (VUAMC/News)
- MRW‑labelled **of** in **News** is de‑counted to mitigate a known annotation artefact.

---

## 2) Multi‑Word Expressions (MWEs)

- TEI anchor/part pairs (`xml:id` ↔ `corresp`) are **merged into a single token**.
- `Original_Word` and `Lemma` are concatenated; POS joins as `POS+POS`.
- `MRW`/`mflag` propagate to the anchor; `type`/`subtype` are **unioned** using `|` with duplicates removed.

---

## 3) Analyses Provided

### 3.1 Single lemma
- Summary metrics (raw/LU/MRW; density per 1,000 LU; share of corpus) plus **two KWIC CSVs** (**MRW** and **non‑MRW**).
- KWIC inserts `[SENT_START]` / `[SENT_END]` sentinels; MRWs are suffixed `_Ω`, capped by `OMEGA_LIMIT` (default **3**).

### 3.2 Batch lemmas
- Executes the single‑lemma pipeline for a list provided inline (comma‑separated) or via text file (one lemma per line).

### 3.3 Collocations (windowed)
- Counts left/right collocates by **distance** (default `COLLOCATION_WINDOW = 5`), aggregates duplicates; exports `lemma, collocate, pos, side, distance, count`.

### 3.4 Pattern search (Regex & CQL)
- **Word‑form** default (enforce with `w:`); **lemma** with `l:`; **raw regex** with `re:`.
- **CQL subset** supported, e.g., `[lemma="gehen"] []{0,1} [pos="NN.*"]`. Saves KWIC CSVs for matches.

### 3.5 Print / Export
- **Full flat CSV** for current scope; **MRW list**; **MRW‑by‑POS** (enter e.g., `NN`).

---

## 4) Data Ingestion

### 4.1 VUAMC
- Auto‑discovers `VUAMC.xml` or a ZIP containing it.
- Parses once **per (path, mtime)** and caches a DataFrame to avoid re‑parsing.

### 4.2 User files (CSV/TSV/XML/ZIP)
- **CSV/TSV:** delimiter auto‑detect; robust encoding attempt (`utf‑8`, `latin1`, `iso‑8859‑1`). Column names are normalized (e.g., `File_ID → file_id`).
  - **Minimal required columns:** `lemma, word, metaphor_function, pos`.
  - **Recommended:** `file_id, sentence_id, type, subtype, mflag, genre`.
- **XML/ZIP:** TEI parsing applies the same MWE merge, masks, and sentence policies as VUAMC.
- **KWIC/collocations** benefit from a valid `sentence_id` for precise boundaries.

---

## 5) Outputs

All exports are written under **`results/`** in a subfolder in the script folder. CSVs are UTF‑8 with BOM for Excel compatibility.

- **Single/Batch KWIC:** dual CSVs (`*_MRW.csv`, `*_NONMRW.csv`) including left/right contexts, node token, lemma/genre, `sentence_id`, and `metaphor_function`.
- **Pattern KWIC:** per‑query CSV with sentence‑level context and the original query.
- **Collocations:** `lemma, collocate, pos, side, distance, count`.
- **Print / Full flat CSV:** `File_ID, Sentence_ID, Original_Word, Lemma, POS, Metaphor, Type, Subtype, MFlag, Genre, xml:id, corresp`.

---

## 6) Configuration & Environment (overview)

- **Environment variables:**

> For platform setup and launch, see **LAUNCHER.md**.

---

## 7) Package Structure (orientation)

- `meta4.cli` — entry menu and user interaction
- `meta4.io` — discovery & loading (CSV/TSV/XML/ZIP), VUAMC auto‑detect & caching
- `meta4.parser` — TEI parsing, sentence handling, MWE merge
- `meta4.analysis` — MRW masks, counts, densities, exports
- `meta4.cql` — CQL/regex machinery and query execution
- `meta4.mipvu` — MIPVU‑specific rules
- `meta4.utils` / `meta4.config` — utilities and configuration

---

## 8) Citations & License

- MIPVU: A Method for Linguistic Metaphor Identification (Vrije Universiteit).
- VU Amsterdam Metaphor Corpus (VUAMC).
- Project license: see repository license declaration.


---
## 9) Author

Developed and maintained as part of ongoing research on metaphor study at the University of Erfurt.

[Daban Q. Jaff] (2025). MeTa4 Metaphor Analysis Tool. Available at: https://github.com/dabjaff/MeTa4-Metaphor-Analysis-Tool
