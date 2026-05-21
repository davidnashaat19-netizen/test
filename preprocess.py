"""
=============================================================================
 preprocess.py  —  IR Text Processing Pipeline
 Course  : CS313x Information Retrieval
 Project : Freelance Market Monitor (Sha8lny Welnaby)
=============================================================================

WHAT THIS SCRIPT DOES (Slide-06 Implementation)
-------------------------------------------------
Reads  : freelance_data.json   (raw scraper output from scraper_ManeualSubmission.py)
Writes : data_before_cleaning.csv  (raw, unformatted data)
         data_after_cleaning.csv   (tokenized, normalized, IR-ready data)

INPUT JSON SCHEMA (written by scraper export_to_json, schema_version 2.0)
--------------------------------------------------------------------------
{
    "metadata": { "total_records": int, "platforms": [...], "scraped_at": str,
                  "schema_version": "2.0", "crawl_type": "deep" },
    "projects": [
        {
            "platform":            str,            e.g. "Freelancer.com" | "Mostaqel.com"
            "title":               str | null,
            "url":                 str | null,
            "budget_min":          float | null,   already parsed by scraper clean_budget()
            "budget_max":          float | null,   already parsed by scraper clean_budget()
            "budget_currency":     str | null,     e.g. "USD" | "SAR" | "EGP" | "GBP"
            "budget_type":         str | null,     "fixed" | "hourly" | "unknown"
            "skills":              list[str],
            "category":            str | null,
            "posted_date":         str | null,
            "full_description":    str | null,     full body from detail page (deep crawl)
            "description_snippet": str | null      card-level teaser (fallback)
        },
        ...
    ]
}

OUTPUT: data_before_cleaning.csv  columns
-----------------------------------------
  platform, title, url, budget_min, budget_max, budget_currency,
  budget_type, skills (pipe-joined), category, posted_date,
  full_description, description_snippet

OUTPUT: data_after_cleaning.csv  columns
-----------------------------------------
  platform, url, title_raw, title_clean, cleaned_text, full_description,
  description_snippet, tokens, token_count, budget_extracted,
  budget_currency, budget_type, skills_clean (pipe-joined),
  category_clean, posted_date

  FIELD DEFINITIONS
  title_clean        — pipeline applied to the TITLE ONLY (no description)
  cleaned_text       — pipeline applied to description + skills + category
                       (title excluded to avoid double-counting in token bag)
  description_snippet — card-level teaser preserved from scraper output
  budget_extracted   — midpoint of budget_min/budget_max; None when both=0
                       (zero means budget was never fetched, not a real value)
  skills_clean       — deduplicated after lowercasing (Freelancer sends the
                       category header as first skill, creating a duplicate)

NLP PIPELINE STEPS
------------------
1. Noise Reduction (Regex)     — strip HTML tags, URLs, special chars
2. Arabic Normalization        — unify Alef/Ta-Marbuta/Waw/Yaa variants
3. English Lowercasing         — all Latin text -> lowercase
4. Stop-word Removal           — remove EN + AR common words (NLTK corpora)
5. Tokenization                — whitespace split on clean text
6. Entity Extraction (Budget)  — budget_min/max are already floats from the
                                  scraper; midpoint is computed directly.
                                  Raw-string fallback only fires when both
                                  values are None/NaN (e.g. "Negotiable").
                                  budget_extracted is stored as None (not 0)
                                  when both budget_min and budget_max are 0.

BUG FIXES APPLIED (relative to original)
-----------------------------------------
Bug 1+2: title_clean is now the cleaned TITLE ONLY. cleaned_text is the
         cleaned description+skills+category only. Previously both were
         title+description concatenated, and cleaned_text began with
         title_clean verbatim — double-counting title tokens.
Bug 3:   skills_clean is deduplicated after lowercasing. Freelancer.com
         prepends the category as the first skill element, producing a
         duplicate once both are lowercased.
Bug 4:   category_clean is still "unknown" for all rows — neither platform
         was scraped for category. A warning is printed in the summary.
Bug 5:   description_snippet is now preserved as its own column instead of
         being silently dropped.
Bug 6:   budget_extracted is stored as None (null) when both budget_min and
         budget_max are 0 — zero means the budget was never fetched, not a
         real $0 project.

HOW TO RUN
----------
    python preprocess.py
    python preprocess.py --input my_data.json --output-dir ./output
"""

import argparse
import json
import re
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download NLTK stopwords corpus (only downloads once; safe to call repeatedly)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------------
# Arabic stop-words — loaded from NLTK's complete corpus (754 words)
# ---------------------------------------------------------------------------
NLTK_ARABIC_STOPWORDS = set(stopwords.words('arabic'))
ARABIC_STOPWORDS = NLTK_ARABIC_STOPWORDS

# ---------------------------------------------------------------------------
# English stop-words — loaded from NLTK's complete corpus (179 words)
# ---------------------------------------------------------------------------
ENGLISH_STOPWORDS = set(stopwords.words('english'))

ALL_STOPWORDS = ARABIC_STOPWORDS | ENGLISH_STOPWORDS

# ---------------------------------------------------------------------------
# Step 1 — Noise Reduction (Regex HTML stripping)
# ---------------------------------------------------------------------------
_HTML_TAG  = re.compile(r"<[^>]+>")
_URL_PAT   = re.compile(r"https?://\S+|www\.\S+")
_SPECIAL   = re.compile(r"[^\w\s\u0600-\u06FF\.\-]", re.UNICODE)
_MULTI_SPC = re.compile(r"\s+")


def remove_noise(text: str) -> str:
    """Strip HTML tags, URLs, and non-alphanumeric characters."""
    if not text or not isinstance(text, str):
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _URL_PAT.sub(" ", text)
    text = _SPECIAL.sub(" ", text)
    text = _MULTI_SPC.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 2 — Arabic Normalization
# ---------------------------------------------------------------------------
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")          # normalize to bare Alef ا
_TA_MARBUTA    = re.compile(r"ة")                # ة -> ه  (for IR matching)
_WAW_VARIANTS  = re.compile(r"ؤ")               # ؤ -> و
_YAA_VARIANTS  = re.compile(r"[ىئ]")            # ى ئ -> ي
_TATWEEL       = re.compile(r"\u0640")           # remove tatweel ـ
_DIACRITICS    = re.compile(r"[\u064B-\u065F]")  # remove tashkeel (harakat)


def normalize_arabic(text: str) -> str:
    """
    Unify Arabic character variants for consistent IR matching.
    Example: 'أحمد' and 'احمد' map to the same token after normalization.
    """
    if not text:
        return text
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _TA_MARBUTA.sub("ه", text)
    text = _WAW_VARIANTS.sub("و", text)
    text = _YAA_VARIANTS.sub("ي", text)
    text = _TATWEEL.sub("", text)
    text = _DIACRITICS.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Step 3 — Lowercasing (English)
# ---------------------------------------------------------------------------
def lowercase_english(text: str) -> str:
    """Lowercase only ASCII/Latin characters; Arabic has no case concept."""
    return text.lower()


# ---------------------------------------------------------------------------
# Step 4 — Stop-word Removal & Tokenization
# ---------------------------------------------------------------------------
def tokenize_and_filter(text: str) -> list:
    """
    Split on whitespace, remove stop-words and single-character tokens.
    Returns a list of meaningful tokens ready for IR indexing.
    """
    tokens = text.split()
    return [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Step 5 — Entity Extraction (Budget)
# ---------------------------------------------------------------------------
# budget_min and budget_max are already parsed to float|None by the scraper's
# clean_budget() function.  The midpoint is therefore a simple arithmetic
# operation.  extract_budget_float() is a string-parsing fallback used only
# when both values are None/NaN (e.g. the project was listed as "Negotiable").
_NUMBER_PAT = re.compile(r"[\d,]+\.?\d*")


def extract_budget_float(raw: str) -> float | None:
    """
    Parse a messy budget string into a single float (midpoint).
    Slide-06 example: 'SR 200 - 500' -> 350.0
    Only called when budget_min and budget_max are both None/NaN.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw_clean = raw.replace(",", "")
    nums = [float(n) for n in _NUMBER_PAT.findall(raw_clean) if n]
    if not nums:
        return None
    return sum(nums) / len(nums)


# ---------------------------------------------------------------------------
# Full pipeline function
# ---------------------------------------------------------------------------
def clean_text_full(text: str) -> tuple[str, list]:
    """
    Run the complete 4-step NLP pipeline on a single text field.
    Returns: (cleaned_text_string, tokens_list)
    """
    t = remove_noise(text)
    t = normalize_arabic(t)
    t = lowercase_english(t)
    tokens = tokenize_and_filter(t)
    return " ".join(tokens), tokens


def safe_str(val) -> str:
    """Safe conversion to string, stripping NaN / None values."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def process_row(row: dict) -> dict:
    """
    Apply the full IR preprocessing pipeline to one project record.

    TEXT FIELD STRATEGY
    -------------------
    title_clean  — cleaned version of the TITLE ONLY (no description mixed in).
    cleaned_text — IR corpus field: description + skills + category.
                   The title is intentionally NOT concatenated here so that
                   title vocabulary is not double-counted in the token bag.

    ALIGNMENT NOTE
    --------------
    The scraper writes budget_min / budget_max as float | None (already parsed
    by clean_budget()).  We therefore compute the midpoint directly from those
    numeric values and only fall back to string parsing when both are None/NaN
    (e.g. "Negotiable" projects where clean_budget returned (None, None, ...)).

    url is preserved in the output so that data_after_cleaning.csv records can
    be traced back to their source page — it is present in data_before_cleaning
    .csv and should not be silently dropped after cleaning.
    """
    # --- Title: clean the title field on its own ---
    title_raw_str = safe_str(row.get("title"))
    title_cleaned, _ = clean_text_full(title_raw_str)

    # --- Description: prefer full_description, fall back to description_snippet ---
    desc = safe_str(row.get("full_description") or row.get("description_snippet") or "")

    skills_list = []
    for s in (row.get("skills") or []):
        s_str = safe_str(s)
        if s_str:
            skills_list.append(s_str)

    # corpus text = description + skills + category (title excluded to avoid double-counting)
    corpus_raw = " ".join(filter(None, [
        desc,
        " ".join(skills_list),
        safe_str(row.get("category")),
    ]))

    cleaned_text, tokens = clean_text_full(corpus_raw)

    # --- Budget entity extraction ---
    # budget_min / budget_max are already floats from the scraper; use them
    # directly.  Fall back to string parsing only when both are absent.
    b_min = row.get("budget_min")
    b_max = row.get("budget_max")

    min_valid = b_min is not None and not (isinstance(b_min, float) and pd.isna(b_min))
    max_valid = b_max is not None and not (isinstance(b_max, float) and pd.isna(b_max))

    if min_valid and max_valid:
        try:
            budget_extracted = (float(b_min) + float(b_max)) / 2
        except (ValueError, TypeError):
            budget_extracted = None
    elif min_valid:
        try:
            budget_extracted = float(b_min)
        except (ValueError, TypeError):
            budget_extracted = None
    elif max_valid:
        try:
            budget_extracted = float(b_max)
        except (ValueError, TypeError):
            budget_extracted = None
    else:
        # Both are None/NaN (e.g. "Negotiable") — attempt string parsing
        budget_raw = (safe_str(b_min) + " " + safe_str(b_max)).strip()
        budget_extracted = extract_budget_float(budget_raw) if budget_raw else None

    # Bug 6 fix: a midpoint of exactly 0.0 means the scraper stored (0, 0)
    # for a project whose budget was never fetched (currency="Unknown").
    # Zero is misleading for downstream analysis — store as None instead.
    if budget_extracted == 0.0:
        budget_extracted = None

    # ── Bug 3 fix: deduplicate skills after lowercasing ──────────────────────
    # Freelancer prepends a category header (e.g. "WEB DEVELOPMENT") as the
    # first skill — identical to the first real skill after lowercasing.
    # We deduplicate case-insensitively while preserving insertion order.
    raw_skills = row.get("skills") or []
    skills_clean_list = []
    _seen_skills: set = set()
    for s in raw_skills:
        s_str = safe_str(s)
        if not s_str:
            continue
        cleaned_skill = " ".join(tokenize_and_filter(
            lowercase_english(normalize_arabic(remove_noise(s_str)))))
        if cleaned_skill and cleaned_skill not in _seen_skills:
            _seen_skills.add(cleaned_skill)
            skills_clean_list.append(cleaned_skill)

    return {
        # ── Traceability ──────────────────────────────────────────────────
        "platform":            row.get("platform"),
        "url":                 safe_str(row.get("url")) or None,
        # ── Title (Bug 1+2 fix: title_clean is the title ONLY) ───────────
        "title_raw":           safe_str(row.get("title")) or "Unknown Title",
        "title_clean":         title_cleaned[:120] if title_cleaned else "",
        # ── Corpus text (description + skills + category; no title) ──────
        "cleaned_text":        cleaned_text,
        "full_description":    desc,
        # ── Bug 5 fix: preserve description_snippet instead of dropping ──
        "description_snippet": safe_str(
            row.get("description_snippet") or ""),
        "tokens":              tokens,
        "token_count":         len(tokens),
        # ── Budget (Bug 6 fix: zero budgets stored as None, not 0) ───────
        "budget_extracted":    budget_extracted if budget_extracted else None,
        "budget_currency":     row.get("budget_currency"),
        "budget_type":         row.get("budget_type"),
        # ── Skills (Bug 3 fix: deduplicated) ─────────────────────────────
        "skills_clean":        skills_clean_list,
        # ── Category (Bug 4: mark as unknown explicitly) ──────────────────
        "category_clean":      lowercase_english(
            normalize_arabic(remove_noise(safe_str(row.get("category"))))),
        # ── Metadata ──────────────────────────────────────────────────────
        "posted_date":         row.get("posted_date"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="IR Text Processing Pipeline — Sha8lny Welnaby")
    parser.add_argument("--input",      default="freelance_data.json",
                        help="Path to raw JSON from scraper_ManeualSubmission.py")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write CSV outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_path = output_dir / "data_before_cleaning.csv"
    after_path  = output_dir / "data_after_cleaning.csv"

    # ── Load JSON ───────────────────────────────────────────────────────────
    if not input_path.exists():
        print(f"ERROR: Cannot find {input_path}. Run scraper_ManeualSubmission.py first.")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # The scraper wraps records under the "projects" key (schema_version 2.0).
    # The top-level "metadata" key is informational only and is not processed.
    projects = data.get("projects", [])
    if not projects:
        print("ERROR: No projects found in JSON.")
        sys.exit(1)

    # ── Deduplicate by url ──────────────────────────────────────────────────
    df_raw = pd.DataFrame(projects)
    if "url" in df_raw.columns:
        before_len = len(df_raw)
        df_raw.drop_duplicates(subset=["url"], keep="first", inplace=True)
        dupes = before_len - len(df_raw)
        if dupes:
            print(f"Deduplicated {dupes} records based on unique url.")
    projects = df_raw.to_dict(orient="records")

    print(f"Loaded {len(projects)} raw records from {input_path}")

    # ── Missing Value Detection & Treatment ─────────────────────────────────
    print("\n" + "=" * 55)
    print("  MISSING VALUE ANALYSIS (Before Cleaning)")
    print("=" * 55)
    df_check = pd.DataFrame(projects)
    missing     = df_check.isnull().sum()
    missing_pct = (missing / len(df_check) * 100).round(1)
    missing_report = missing[missing > 0]
    if missing_report.empty:
        print("  ✔ No missing values detected.")
    else:
        print(f"  {'Column':<30} {'Missing':>8} {'% Missing':>10}")
        print(f"  {'-' * 50}")
        for col in missing_report.index:
            print(f"  {col:<30} {missing[col]:>8} {missing_pct[col]:>9}%")

    # Treatment strategy:
    # 1. Drop rows where BOTH title AND full_description are null (useless for IR)
    df_check.dropna(subset=["title", "full_description"], how="all", inplace=True)

    # 2. Fill remaining text columns with 'Unknown'
    text_cols = ["title", "full_description", "description_snippet", "category"]
    for col in text_cols:
        if col in df_check.columns:
            df_check[col] = df_check[col].fillna("Unknown")

    # 3. Numeric budget columns: the scraper already writes float | None.
    #    Impute missing values using the median for that currency group so
    #    that budget_extracted in the cleaned CSV is always numeric.
    if "budget_currency" not in df_check.columns:
        df_check["budget_currency"] = "Unknown"

    for col in ["budget_min", "budget_max"]:
        if col in df_check.columns:
            df_check[col] = pd.to_numeric(df_check[col], errors="coerce")
            df_check[col] = df_check.groupby("budget_currency")[col].transform(
                lambda x: x.fillna(x.median() if not x.dropna().empty else 0)
            )
            df_check[col] = df_check[col].fillna(0)   # fallback if group entirely null

    # 4. Ensure skills is always a list
    if "skills" in df_check.columns:
        df_check["skills"] = df_check["skills"].apply(
            lambda x: x if isinstance(x, list) else [])

    # 5. Fill remaining categorical / metadata columns
    for col in ["platform", "url", "posted_date", "budget_currency", "budget_type"]:
        if col in df_check.columns:
            df_check[col] = df_check[col].fillna("Unknown")

    remaining = df_check.isnull().sum().sum()
    print(f"\n  Missing values after treatment: {remaining}")
    print("=" * 55)

    projects = df_check.to_dict(orient="records")

    # ── BEFORE: raw data CSV ─────────────────────────────────────────────────
    # Columns mirror the scraper's FreelanceProject dataclass fields exactly:
    #   platform, title, url, budget_min, budget_max, budget_currency,
    #   budget_type, skills, category, posted_date, full_description,
    #   description_snippet
    df_before = pd.DataFrame(projects)
    if "skills" in df_before.columns:
        df_before["skills"] = df_before["skills"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else str(x or ""))
    df_before.to_csv(before_path, index=False, encoding="utf-8-sig")
    print(f"[BEFORE] data_before_cleaning.csv  -> {len(df_before)} rows, "
          f"{len(df_before.columns)} columns saved to {before_path}")

    # ── Apply IR pipeline ────────────────────────────────────────────────────
    print("\nRunning IR preprocessing pipeline...")
    processed = []
    for i, row in enumerate(projects):
        try:
            processed.append(process_row(row))
        except Exception as exc:
            print(f"  WARNING: Row {i} failed: {exc}")

    # ── AFTER: cleaned IR-ready CSV ──────────────────────────────────────────
    # Columns: platform, url, title_raw, title_clean, cleaned_text,
    #          full_description, description_snippet, tokens, token_count,
    #          budget_extracted, budget_currency, budget_type,
    #          skills_clean, category_clean, posted_date
    df_after = pd.DataFrame(processed)
    if "tokens" in df_after.columns:
        df_after["tokens"] = df_after["tokens"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else str(x or ""))
    if "skills_clean" in df_after.columns:
        df_after["skills_clean"] = df_after["skills_clean"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else str(x or ""))
    df_after.to_csv(after_path, index=False, encoding="utf-8-sig")
    print(f"[AFTER]  data_after_cleaning.csv   -> {len(df_after)} rows, "
          f"{len(df_after.columns)} columns saved to {after_path}")

    # ── Summary stats ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PREPROCESSING SUMMARY")
    print("=" * 55)
    print(f"  Total records processed : {len(processed)}")
    if "token_count" in df_after.columns:
        print(f"  Avg tokens per record   : {df_after['token_count'].mean():.1f}")
    if "budget_extracted" in df_after.columns:
        n_budgets = df_after["budget_extracted"].notna().sum()
        n_null    = df_after["budget_extracted"].isna().sum()
        print(f"  Records with budget     : {n_budgets} "
              f"({n_budgets / len(df_after) * 100:.0f}%)")
        if n_null:
            print(f"  ⚠  Budget=None rows     : {n_null} "
                  f"(budget was 0 or unparseable — stored as null, not 0)")
    if "category_clean" in df_after.columns:
        unknown_cats = (df_after["category_clean"].isin(["unknown", ""])).sum()
        if unknown_cats == len(df_after):
            print(f"  ⚠  category_clean       : all {unknown_cats} rows are 'unknown' "
                  f"— category was never scraped from either platform")
    print("=" * 55)

    # ── Sample transformation ────────────────────────────────────────────────
    print("\nSample transformation (first record):")
    if projects and processed:
        raw_sample   = projects[0]
        clean_sample = processed[0]
        print(f"  RAW title    : {raw_sample.get('title', '')[:80]}")
        print(f"  RAW url      : {raw_sample.get('url', '')[:80]}")
        print(f"  CLEAN tokens : {str(clean_sample['tokens'])[:80]}")
        print(f"  Budget       : {raw_sample.get('budget_min')} - "
              f"{raw_sample.get('budget_max')}  ->  {clean_sample['budget_extracted']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
