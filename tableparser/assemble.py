"""Turn a cell grid (from either path) into the final table:
header rows + header paths, row labels, parsed values, OCR cross-check flags.

Challenges: 13-19 (header rows/paths), 23/24 (blank cells as records),
36 (per-table number vote), 41-44 (second-reader disagreement flags),
47-49 (label cleanup, umlaut restore), 50-53 (raw kept, nothing "corrected").
"""
import difflib
import re

from .values import parse_cell, resolve_table

YEAR = re.compile(r"^(19|20)\d{2}$")
PERIOD = re.compile(r"^(Q[1-4]|FY|H[12]|Jan|Feb|M[aä]r|Apr|Ma[iy]|Jun|Jul|Aug|Sep|O[ck]t|Nov|De[cz])\.?$", re.I)

# German words whose umlauts OCR tends to drop (challenge 47)
GERMAN_TERMS = ["Mär", "Umsatzerlöse", "Jahresüberschuss", "Vorräte", "Rückstellungen",
                "Erlöse", "Aufwände", "Gewinnrücklagen", "Steuerrückstellungen", "Überschuss",
                "Zinsaufwand", "Rohertrag", "Steuern", "Abschreibungen", "Forderungen",
                "Verbindlichkeiten", "Eigenkapital", "Materialaufwand", "Personalaufwand",
                "Sonstige Kosten", "Bilanzsumme", "Anlagevermögen", "Umlaufvermögen"]
_DEUML = str.maketrans("äöüÄÖÜ", "aouAOU")
_UMLAUT_FIX = {t.translate(_DEUML).lower(): t for t in GERMAN_TERMS if t.translate(_DEUML) != t}


MATCH_CUTOFF = 0.8        # fuzzy header / label matching
MAX_LEN_DIFF = 0.3        # ... and the match may differ in length by at most 30 %
_GERMAN_EXACT = {t.lower() for t in GERMAN_TERMS}


def _close(t, choices):
    """Best fuzzy match with cutoff MATCH_CUTOFF and length difference <= 30 %."""
    for m in difflib.get_close_matches(t, choices, n=3, cutoff=MATCH_CUTOFF):
        if abs(len(m) - len(t)) <= MAX_LEN_DIFF * len(m):
            return m
    return None


def german_hint(texts):
    """True if the table itself shows German: umlauts/ß, German month names,
    or a GERMAN_TERMS word as printed. Umlaut restoration needs this hint, so
    English labels are never 'restored' into German."""
    for t in texts:
        if not t:
            continue
        if re.search(r"[äöüÄÖÜß]", t) or re.search(r"\b(Mär|Mai|Okt|Dez)\b", t):
            return True
        if re.sub(r"\s+", " ", t).strip(" .,:;-_").lower() in _GERMAN_EXACT:
            return True
    return False


def clean_label(t, german=True):
    if not t:
        return ""
    t = re.sub(r"(?<=[A-Za-zÄÖÜäöüß])[.:,](?=[A-Za-zÄÖÜäöüß])", " ", t)   # "Variant.B"
    t = re.sub(r"[^\w()&%/\- äöüÄÖÜß]", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .,:;-_")
    if not german:
        return t
    fixed = _UMLAUT_FIX.get(t.lower())
    if fixed:
        return fixed
    if len(t) >= 6:                        # OCR-damaged German term, e.g. "Jahrestiberschuss"
        m = _close(t, GERMAN_TERMS)
        if m:
            return m
    return t


# words this kind of table uses in headers; OCR-damaged header words are snapped
# to the closest one (ch. 13-19). Kept small and generic on purpose.
HEADER_VOCAB = ["Q1", "Q2", "Q3", "Q4", "FY", "Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul",
                "Aug", "Sep", "Okt", "Nov", "Dez", "Train", "Dev", "Test", "OOD", "Avg", "Rank",
                "Min", "Max", "Mean", "Median", "Std", "N", "Plan", "Actual", "Delta", "Reported",
                "Adjusted", "Metric", "Item", "Position"]


def snap_header(t, german=True):
    if not t or t in HEADER_VOCAB or YEAR.match(t):
        return t, False
    q = re.fullmatch(r"[QqOo0aä]([1-4iIl|])", t)          # "Qi", "a2", "Ql" -> Q1/Q2
    if q:
        d = q.group(1)
        return "Q" + ("1" if d in "iIl|" else d), True
    if german:                                             # "Mar" -> "Mär" (umlaut restore)
        folded = {h.translate(_DEUML).lower(): h for h in HEADER_VOCAB if h.translate(_DEUML) != h}
        if t.translate(_DEUML).lower() in folded:
            return folded[t.translate(_DEUML).lower()], True
    low = {h.lower(): h for h in HEADER_VOCAB}
    m = _close(t.lower(), list(low))
    if m and not german and low[m] != low[m].translate(_DEUML):
        m = None                                           # no umlaut words without a German hint
    return (low[m], True) if m else (t, False)


NA_GUESS = re.compile(r"^(w|n|m|o|ni|nl|ri)?[/l1|]?[ae]$|^oma$|^nia$", re.I)


def is_real_number(text):
    """True for a body value. Years, Q1, FY, month names are header material."""
    t = (text or "").strip()
    if not t or YEAR.match(t) or PERIOD.match(t):
        return False
    p = parse_cell(t)
    return p["kind"] == "number" and p["value"] != "UNPARSED"


def _drop_empty(cells, R, C):
    used_r = {r for c in cells if c["text"] for r in range(c["r0"], c["r1"] + 1)}
    used_c = {k for c in cells if c["text"] for k in range(c["c0"], c["c1"] + 1)}
    rmap = {r: i for i, r in enumerate(sorted(used_r))}
    cmap = {k: i for i, k in enumerate(sorted(used_c))}
    out = []
    for c in cells:
        rs = [rmap[r] for r in range(c["r0"], c["r1"] + 1) if r in rmap]
        cs = [cmap[k] for k in range(c["c0"], c["c1"] + 1) if k in cmap]
        if not rs or not cs:
            continue
        out.append({**c, "r0": min(rs), "r1": max(rs), "c0": min(cs), "c1": max(cs)})
    return out, len(rmap), len(cmap)


def assemble(cells, R, C, meta=None):
    cells, R, C = _drop_empty(cells, R, C)
    at = {}
    for i, c in enumerate(cells):
        for r in range(c["r0"], c["r1"] + 1):
            for k in range(c["c0"], c["c1"] + 1):
                at[(r, k)] = i

    # --- header rows: top rows whose value cells hold no real number (ch. 18/19)
    n_header = 0
    for r in range(R):
        vals = {at[(r, k)] for k in range(1, C) if (r, k) in at}
        if any(is_real_number(cells[i]["text"]) for i in vals):
            break
        n_header += 1
    if n_header == R:          # no numbers anywhere: assume one header row
        n_header = 1

    german = german_hint([c["text"] for c in cells])

    # --- header path per value column (ch. 13-16)
    columns, snapped_headers = [], []
    for k in range(1, C):
        path, seen = [], set()
        for r in range(n_header):
            i = at.get((r, k))
            if i is None or i in seen:
                continue
            seen.add(i)
            t, snapped = snap_header(clean_label(cells[i]["text"], german), german)
            if t:
                path.append(t)
                if snapped:
                    snapped_headers.append(t)
        columns.append(path or [f"col{k}"])
    label_header = " ".join(dict.fromkeys(
        snap_header(clean_label(cells[at[(r, 0)]]["text"], german), german)[0] for r in range(n_header)
        if (r, 0) in at and cells[at[(r, 0)]]["text"])) or "label"

    # --- body
    body_rows = list(range(n_header, R))
    labels = {r: clean_label(cells[at[(r, 0)]]["text"], german) if (r, 0) in at else "" for r in body_rows}
    value_cells = sorted({at[(r, k)] for r in body_rows for k in range(1, C) if (r, k) in at},
                         key=lambda i: (cells[i]["r0"], cells[i]["c0"]))
    for i in value_cells:                       # trailing specks: "56.408,29 kg." (ch. 42)
        cells[i]["text"] = re.sub(r"(?<=[\w%€)])[.:;,'`]+$", "", cells[i]["text"].strip())
        if cells[i]["text"].lower() != "n/a" and NA_GUESS.match(cells[i]["text"]):   # "Wa", "oma" (ch. 23)
            cells[i]["na_guess"], cells[i]["text"] = True, "n/a"
    texts = [cells[i]["text"] for i in value_cells if cells[i]["text"]]
    parsed, decimals = resolve_table(texts) if texts else ([], None)
    by_text_idx = iter(parsed)

    records = []
    for i in value_cells:
        c = cells[i]
        rec = {"row": labels.get(c["r0"], ""), "row_index": c["r0"] - n_header,
               "cols": [columns[k - 1] for k in range(max(1, c["c0"]), c["c1"] + 1)],
               "col_index": list(range(max(1, c["c0"]) - 1, c["c1"])),
               "raw": c["text"], "ocr_conf": round(c.get("conf", 0.0), 1)}
        if not c["text"]:
            rec.update({"kind": "empty", "value": None, "unit": None, "flags": ["blank"]})
        else:
            p = next(by_text_idx)
            flags = [p["flag"]] if p["flag"] != "ok" else []
            # second reader (ch. 41-44): use it if the first read failed, flag if they disagree
            alt = c.get("alt")
            if alt:
                pa = parse_cell(alt["text"])
                if p["kind"] != "number" or p["value"] == "UNPARSED":
                    if pa["kind"] == "number" and pa["value"] != "UNPARSED":
                        p, flags = pa, ["second_reader_used"]
                        rec["raw"] = alt["text"]
                elif pa["kind"] == "number" and pa["value"] not in ("UNPARSED", p["value"]):
                    clean = ("ok", "ok_by_table", "digit_fixed")
                    if p.get("orig_flag", p["flag"]) not in clean and pa["flag"] in clean:
                        # second read is clean, first needed repair -> trust the clean one
                        rec["alt_value"], p = p["value"], pa
                        rec["raw"] = alt["text"]
                        flags = ["readers_disagree"]
                    else:
                        flags.append("readers_disagree")
                        rec["alt_value"] = pa["value"]
            if p.get("repaired_text") and rec["raw"] == c["text"]:
                # rebuilt from digits in the table's format; raw stays the OCR read
                rec["repaired_raw"] = p["repaired_text"]
            if c.get("sign_doubt"):
                flags.append("check_sign")
            if c.get("na_guess"):
                flags.append("marker_guess")
            if c.get("conf", 100) < 50 and p["kind"] == "number":
                flags.append("low_ocr_confidence")
            rec.update({"kind": p["kind"], "value": p["value"], "unit": p.get("unit"),
                        "flags": flags})
            if p["kind"] == "text":
                rec["text"] = p.get("text", c["text"])
        if len(rec["cols"]) > 1:
            rec["merged"] = True
        records.append(rec)

    table = {"label_header": label_header, "columns": columns,
             "rows": [labels[r] for r in body_rows], "cells": records,
             "meta": {**(meta or {}), "header_rows": n_header, "decimals_vote": decimals,
                      "headers_snapped": snapped_headers,
                      "n_flagged": sum(1 for r in records
                                      if [f for f in r["flags"] if f not in ("blank", "ok_by_table")])}}
    for r in records:
        r["_decimals"] = decimals
    table["markdown"] = to_markdown(table)
    for r in records:
        r.pop("_decimals", None)
    return table


def _fmt(rec):
    if rec["kind"] == "number":
        if rec["value"] == "UNPARSED":
            s = f"?{rec['raw']}?"
        else:
            v = rec["value"]
            d = rec.get("_decimals")
            s = f"{v:.{d}f}" if d is not None else (f"{v:.2f}" if not float(v).is_integer() else f"{int(v)}")
            if rec.get("unit"):
                s += f" {rec['unit']}"
    elif rec["kind"] == "text":
        s = rec.get("text") or rec["raw"]
    else:
        s = rec["raw"] if rec["raw"] else ""
    real_flags = [f for f in rec["flags"] if f not in ("blank", "ok_by_table")]
    return s + (" ⚠" if real_flags else "")


def to_markdown(table):
    cols = table["columns"]
    head = [table["label_header"]] + [" / ".join(p) for p in cols]
    grid = [[lab] + [""] * len(cols) for lab in table["rows"]]
    for rec in table["cells"]:
        for k in rec["col_index"]:
            if 0 <= rec["row_index"] < len(grid):
                grid[rec["row_index"]][k + 1] = _fmt(rec)   # merged: repeated in each column
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(row) + " |" for row in grid]
    return "\n".join(lines)
