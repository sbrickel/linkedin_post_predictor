# process_linkedin_data.py
import pandas as pd
import glob
import os
import warnings
import re

warnings.simplefilter("ignore")  # suppress openpyxl style warnings

URL_RE = re.compile(r"https?://(?:www\.)?linkedin\.com[^\s]*", flags=re.IGNORECASE)

def parse_int(value):
    """Safely convert a cell to int (handles commas, NaN, strings)."""
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    try:
        return int(value)
    except Exception:
        try:
            # sometimes floats are present
            return int(float(value))
        except Exception:
            return 0

def looks_like_url(val):
    if pd.isna(val):
        return False
    s = str(val).strip()
    return bool(URL_RE.search(s))

def find_any_linkedin_url_in_df(df_raw):
    # search whole sheet for a linkedin url
    for i in range(df_raw.shape[0]):
        for j in range(df_raw.shape[1]):
            cell = df_raw.iat[i, j]
            if looks_like_url(cell):
                return str(cell).strip()
    return None

def normalize_key(s):
    return str(s).strip().lower() if pd.notna(s) else ""

def extract_metrics_from_performance(filepath):
    """
    Attempt multiple strategies to extract a key->value mapping from the PERFORMANCE sheet.
    Returns a dict of normalized keys to values and the raw DataFrame used.
    """
    # Read twice: once with header=None (raw), once default (may parse header row)
    try:
        df_raw = pd.read_excel(filepath, sheet_name="PERFORMANCE", header=None)
    except Exception as e:
        # Some files have different sheet naming/casing; try fallback
        df_raw = pd.read_excel(filepath, sheet_name=0, header=None)

    # Strategy A: if the sheet parsed with header row (first row looks like header names),
    # read again with header=0 and treat columns as keys with their first non-null value.
    first_row_strings = df_raw.iloc[0].astype(str).str.lower().fillna("")
    # detect if likely a horizontal header by checking for key names in first row
    header_like = any(x for x in first_row_strings if any(k in x for k in ["post url", "post date", "impressions", "reactions", "comments"]))
    metrics = {}

    if header_like:
        # read with header=0 (pandas will use first row as column headers)
        try:
            df_h = pd.read_excel(filepath, sheet_name="PERFORMANCE", header=0)
        except Exception:
            df_h = pd.read_excel(filepath, sheet_name=0, header=0)

        # collect first non-null value for each column (most exports put values in the first data row)
        for col in df_h.columns:
            colkey = normalize_key(col)
            # take first non-null cell in this column
            try:
                val = df_h[col].dropna().iloc[0]
            except Exception:
                val = None
            metrics[colkey] = val

        # quick attempt to extract URL
        if "post url" in metrics and looks_like_url(metrics.get("post url")):
            return metrics, df_raw

    # Strategy B: vertical key-value pairs (common for LinkedIn "Performance" sheet)
    # treat column 0 as key and column 1 as value where present
    if df_raw.shape[1] >= 2:
        keys = df_raw.iloc[:, 0].astype(str).fillna("").str.strip()
        vals = df_raw.iloc[:, 1]
        for k, v in zip(keys, vals):
            nk = normalize_key(k)
            if nk != "" and nk not in metrics:
                metrics[nk] = v

    # Strategy C: Some files have key in a cell, value in the cell below or to the right
    # scan for target keys and pull neighbor cells
    target_keys = ["post url", "posturl", "post link", "url"]
    for i in range(df_raw.shape[0]):
        for j in range(df_raw.shape[1]):
            cell = df_raw.iat[i, j]
            if isinstance(cell, str) and any(k in cell.lower() for k in target_keys):
                # try right cell
                cand = None
                if j + 1 < df_raw.shape[1]:
                    cand = df_raw.iat[i, j + 1]
                    if looks_like_url(cand):
                        metrics["post url"] = str(cand).strip()
                        return metrics, df_raw
                # try below cell
                if i + 1 < df_raw.shape[0]:
                    cand = df_raw.iat[i + 1, j]
                    if looks_like_url(cand):
                        metrics["post url"] = str(cand).strip()
                        return metrics, df_raw
                # try diagonal
                if i + 1 < df_raw.shape[0] and j + 1 < df_raw.shape[1]:
                    cand = df_raw.iat[i + 1, j + 1]
                    if looks_like_url(cand):
                        metrics["post url"] = str(cand).strip()
                        return metrics, df_raw

    # Strategy D: fallback — search entire sheet for any linkedin url
    url_found = find_any_linkedin_url_in_df(df_raw)
    if url_found:
        metrics["post url"] = url_found

    return metrics, df_raw

def process_performance_sheet(filepath: str):
    """Return a single-row DataFrame with extracted features from the PERFORMANCE sheet."""
    metrics, df_raw = extract_metrics_from_performance(filepath)

    # extract fields safely using normalized keys
    post_url = metrics.get("post url") or metrics.get("posturl") or metrics.get("post link") or None
    # attempt to coerce post_date from several possible key names
    post_date_candidates = [metrics.get(k) for k in ("post date", "date", "published", "post_date") if k in metrics]
    post_date = None
    for cand in post_date_candidates:
        try:
            post_date = pd.to_datetime(cand, errors="coerce")
            if pd.notna(post_date):
                break
        except Exception:
            post_date = None

    post_time = metrics.get("post publish time") or metrics.get("publish time") or metrics.get("time") or ""

    impressions = parse_int(metrics.get("impressions", metrics.get("views", 0)))
    reactions = parse_int(metrics.get("reactions", 0))
    comments = parse_int(metrics.get("comments", 0))
    reposts = parse_int(metrics.get("reposts", 0))
    saves = parse_int(metrics.get("saves", 0))

    weekday = post_date.weekday() if pd.notnull(post_date) else None
    month = post_date.month if pd.notnull(post_date) else None
    year = post_date.year if pd.notnull(post_date) else None

    # Extract hour heuristically
    hour = None
    if isinstance(post_time, str) and post_time.strip():
        try:
            hour = int(post_time.split(":")[0])
        except Exception:
            # try to parse if it's a timestamp
            try:
                parsed = pd.to_datetime(post_time, errors="coerce")
                if pd.notnull(parsed):
                    hour = parsed.hour
            except Exception:
                hour = None
    else:
        # sometimes post_time is absent but post_date contains time
        if pd.notnull(post_date) and hasattr(post_date, "hour"):
            try:
                hour = int(post_date.hour)
            except Exception:
                hour = None

    engagement_rate = (reactions + comments + reposts) / impressions if impressions > 0 else 0

    return pd.DataFrame([{
        "post_url": post_url,
        "post_date": post_date,
        "weekday": weekday,
        "month": month,
        "year": year,
        "hour": hour,
        "impressions": impressions,
        "reactions": reactions,
        "comments": comments,
        "reposts": reposts,
        "saves": saves,
        "engagement_rate": engagement_rate
    }])

def consolidate_linkedin_files(input_folder: str, output_file: str):
    all_files = glob.glob(os.path.join(input_folder, "PostAnalytics_SebastianBrickel_*.xlsx"))
    dfs = []
    for file in all_files:
        try:
            df_post = process_performance_sheet(file)
            # simple log about URL presence
            url_val = df_post.at[0, "post_url"]
            print(f"Processed: {os.path.basename(file)} -> post_url={'FOUND' if url_val else 'MISSING'}")
            dfs.append(df_post)
        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        # Ensure post_url column is string dtype (avoid object mixing)
        if "post_url" in df_all.columns:
            df_all["post_url"] = df_all["post_url"].astype("string")
        df_all.to_csv(output_file, index=False)
        print(f"✅ Consolidated {len(dfs)} posts into {output_file}")
    else:
        print("⚠️ No files processed.")

if __name__ == "__main__":
    consolidate_linkedin_files(input_folder="data/raw", output_file="data/linkedin_posts_clean.csv")

