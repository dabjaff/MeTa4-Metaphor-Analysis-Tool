# MeTa4 — Launch & Usage Guide (Installation • Data Placement • Keys • Windows Behaviour)

This guide provides practical, platform‑specific instructions for **installing prerequisites**, **placing data**, and **launching MeTa4**.

---

## 1) Contents of the Release

- `meta4/` (do not move/rename)
- `Start MeTa4.command` (macOS)
- Start MeTa4.bat` (Windows)
- Start MeTa4.sh` (Linux/macOS alternative)
- `run_meta4.py` (Python launcher)

**VUAMC:** Obtain `VUAMC.xml` from the official source (only the XML is required). Place `VUAMC.xml` (or a ZIP that contains it) **in the same folder** as the launchers, or set `METAFY_VUAMC` to its full path.

---

## 2) Python & Dependencies 

Install **for your user account** (no admin rights required).

- **macOS / Linux**
  ```bash
  python3 -m pip install --upgrade pip setuptools wheel --user
  python3 -m pip install "pandas>=2.1,<3" "lxml>=5.1,<6" --user
  ```
- **Windows (PowerShell)**
  ```powershell
  py -m pip install --upgrade pip setuptools wheel --user
  py -m pip install "pandas>=2.1,<3" "lxml>=5.1,<6" --user
  # If 'py' is unavailable, use 'python' instead of 'py'
  ```

### Installing Python 3.10+

- **macOS:** Install from python.org, then verify:
  ```bash
  python3 -V && python3 -m pip --version
  ```
  If `lxml` build errors occur later:
  ```bash
  xcode-select --install
  ```

- **Windows:** Install from python.org and enable **Add Python to PATH**, then verify:
  ```powershell
  py -V; py -m pip --version
  ```
  If `py` is unavailable, try `python -V` and `python -m pip --version`.

- **Linux:** Ensure `python3 >= 3.10` and `pip` are installed via the distribution packages.

---

## 3) Launching

**the launchers**
- macOS: double‑click `Start MeTa4.command` (→ Open if blocked).
- Windows: double‑click `Start MeTa4.bat` (→ Run anyway).
- Linux: `chmod +x "Start MeTa4.sh"` then `./Start MeTa4.sh`.

**Terminal alternative**:
- macOS/Linux:
  ```bash
  cd "/path/to/unzipped/folder"
  python3 -m meta4.cli
  ```
- Windows (PowerShell):
  ```powershell
  cd "C:\path\to\unzipped\folder"
  py -m meta4.cli
  # Or: python -m meta4.cli
  ```

---

## 4) Screen Behaviour

On Windows, the launcher show a **welcome** window and start MeTa4 in a separate **Active Session** console. 

**Welcome / Main Menu (first window)**
```text
Welcome to MeTa4: Metaphor Analysis Tool
Analyze metaphor usage in text corpora using MIPVU methodology.

Main Menu
What to do?
  [1] Explore the VUAMC
  [2] Genre-Level Analysis
  [3] Upload Your File
  [T] Tutorial
  [Q] Quit
Choice:
```

**VUAMC Analysis Menu (second window, after selecting [1])**
```text
VUAMC Analysis Menu
  [1] Single Lemma Analysis
  [2] Batch Lemma Analysis
  [3] Collocation Analysis, Lemma
  [4] Pattern Search (regex/CQL), Word form (pre-normalization)
  [5] Print (export CSVs)
  [H] Help (query power)
  [T] Tutorial (beginner guide)
  [R] Return to Main Menu
  [Q] Quit
Choice:
```

If a console **opens and closes immediately** (e.g., after choosing [1]/[2]/[3]), run from **PowerShell** so the console remains open:
```powershell
cd "C:\path\to\unzipped\folder"
py -m meta4.cli
```

---

## 5) Data Placement & Supported Formats

- **VUAMC:** Download `VUAMC.xml` from: https://llds.ling-phil.ox.ac.uk/llds/xmlui/handle/20.500.14106/2541 
make sure only the xml file(the rest of the files downloaded are not required) is placed in the same folder.

- **User data:** CSV/TSV/XML/ZIP accepted.  
  **Minimal columns (case‑insensitive):** `lemma, word, metaphor_function, pos`.  
  **Recommended:** `file_id, sentence_id, type, subtype, mflag, genre`.

---

## 6) Universal Keys & Interaction

- **[1] [2] [3] [4] …** select menu items  
- **[H]** contextual help; **[T]** tutorial  
- **[R]** return/back; **[Q]** quit  
- **Input:** press Enter to submit values; paste comma‑separated lists for batch lemmas (e.g., `time, run, light`).  
- **Pattern syntax:** word‑form default; `w:` (word), `l:` (lemma), `re:` (regex); CQL examples like `[lemma="gehen"] []{0,1} [pos="NN.*"]`.  
- **Copy/paste:** macOS (⌘C/⌘V); Windows PowerShell (Ctrl+C/Ctrl+V). Quote paths with spaces.

---

## 7) Results

By default, outputs are saved under:
```
.../results/<timestamped-subfolder>/
```
The exact path is printed at the end of each task.

To override the target directory:
- macOS/Linux:
  ```bash
  export META4_RESULTS_DIR="/full/path/for/results"
  ```
- Windows:
  ```powershell
  $env:META4_RESULTS_DIR="C:\full\path\for\results"
  ```


---

## 8) Support

Inside MeTa4: press **[H]** for help or **[T]** for the tutorial.  
For assistance, include the exact command you ran and the complete error text.

---

## 9) Author

Developed and maintained as part of ongoing research on metaphor study at the University of Erfurt.

[Daban Q. Jaff] (2025). MeTa4 Metaphor Analysis Tool. Available at: https://github.com/dabjaff/MeTa4-Metaphor-Analysis-Tool

