from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import List
import hashlib,json,shutil
import pandas as pd
from .schemas import CANON_COL_MAP,REQUIRED_FOR_MIPVU
@dataclass
class UploadManifest:
    original_path:str
    stored_path:str
    sha256:str
    size_bytes:int
    filetype:str
    encoding:str|None
    separator:str|None
    rows:int
    cols:int
    notes:List[str]
def _sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()
def _detect_encoding(path:Path)->str:
    for enc in ("utf-8","utf-8-sig","cp1252","latin1"):
        try:
            with path.open("r",encoding=enc,errors="strict") as f:
                f.read(65536)
            return enc
        except Exception:
            continue
    return "utf-8"
def _infer_sep(sample:str)->str|None:
    counts={",":sample.count(","), "\t":sample.count("\t"), ";":sample.count(";")}
    sep=max(counts,key=counts.get)
    return sep if counts[sep]>0 else None
def _read_user_table(path:Path)->tuple[pd.DataFrame,str,str|None,str|None]:
    suf=path.suffix.lower()
    if suf in {".xlsx",".xls"}:
        df=pd.read_excel(path)
        return df,"excel",None,None
    if suf in {".csv",".tsv",".txt"}:
        enc=_detect_encoding(path)
        with path.open("r",encoding=enc,errors="replace") as f:
            sample=f.read(8192)
        sep="\t" if suf==".tsv" else _infer_sep(sample) or ","
        df=pd.read_csv(path,sep=sep,encoding=enc,engine="python")
        return df,"csv",sep,enc
    raise RuntimeError(f"Unsupported file type: {suf}")
def _normalize_columns(df:pd.DataFrame)->pd.DataFrame:
    ren={}
    for c in df.columns:
        key=str(c).strip()
        key_lower=key.lower()
        ren[c]=CANON_COL_MAP.get(key,CANON_COL_MAP.get(key_lower,key_lower))
    out=df.rename(columns=ren).copy()
    for c in ("lemma","pos","metaphor_function","type","subtype","mflag","genre"):
        if c in out.columns:
            out[c]=out[c].astype(str).str.strip()
    if "word" not in out.columns and "token" in out.columns:
        out["word"]=out["token"].astype(str)
    if "metaphor_function" in out.columns and "mrw" not in out.columns:
        out["mrw"]=out["metaphor_function"].astype(str).str.upper().eq("MRW")
    return out
def _validate(df:pd.DataFrame)->list[str]:
    miss=sorted(list(REQUIRED_FOR_MIPVU-set(df.columns)))
    return [f"Missing required columns for MIPVU counts: {miss} (need {sorted(REQUIRED_FOR_MIPVU)})"] if miss else []
def _target_dir()->Path:
    here=Path(__file__).resolve().parent
    dest=here.parent.joinpath("results","uploads")
    dest.mkdir(parents=True,exist_ok=True)
    return dest
def ingest_user_file(path:str|Path,copy_into:str|Path|None=None)->tuple[UploadManifest,pd.DataFrame]:
    src=Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    df,ftype,sep,enc=_read_user_table(src)
    df=_normalize_columns(df)
    notes=_validate(df)
    dest_root=Path(copy_into) if copy_into else _target_dir()
    from datetime import datetime
    stamp=datetime.now().strftime("%Y%m%d-%H%M%S")
    dest_dir=dest_root/stamp
    dest_dir.mkdir(parents=True,exist_ok=True)
    dest_file=dest_dir/src.name
    shutil.copy2(src,dest_file)
    man=UploadManifest(original_path=str(src),stored_path=str(dest_file),sha256=_sha256(src),size_bytes=src.stat().st_size,filetype=ftype,encoding=enc,separator=sep,rows=int(df.shape[0]),cols=int(df.shape[1]),notes=notes)
    with (dest_dir/"manifest.json").open("w",encoding="utf-8") as f:
        json.dump(asdict(man),f,ensure_ascii=False,indent=2)
    try:
        df.head(50).to_csv(dest_dir/"preview.csv",index=False)
    except Exception:
        pass
    return man,df


def ingest_user_file_df(path, copy_into=None):
    man, df = ingest_user_file(path, copy_into)
    return df
