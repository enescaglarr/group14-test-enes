import re

NA = {"n/a", "na", "-", "–", "—", ""}

KNOWN_UNITS = {"%", "€", "$", "EUR", "USD", "kg", "ms", "g", "t", "m", "km", "h", "s"}
UNIT_FIXES = {"rns": "ms", "m5": "ms", "rn5": "ms", "k9": "kg", "kq": "kg", "Kg": "kg", "o/o": "%"}

def split_unit(s: str):
    m = re.fullmatch(r"(.*?[\d)])\s*(%|€|\$|[A-Za-z]{1,4})", s.strip())
    return (m.group(1), m.group(2)) if m else (s, None)

NUM = re.compile(r"(\d{1,3}(?:[., ]\d{3})*|\d+)(?:([.,])(\d{2}))?")

CONFUSABLE = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1",
                            "S": "5", "B": "8", "Z": "2"})

def fix_digit_confusions(s: str):
    """Inside number-like words only (has a digit, rest digits/separators/confusables),
    map OCR letter look-alikes back to digits: 'O76' -> '076'.
    Words without a digit ('note', 'ms', 'kg', 'n/a', 'Sulfur') are left alone."""
    out, changed = [], False
    for w in s.split(" "):
        if (re.search(r"\d", w) and re.fullmatch(r"[\d.,:()\-OoDlI|SBZ]+", w)
                and re.search(r"[OoDlI|SBZ]", w)):
            w, changed = w.translate(CONFUSABLE), True
        out.append(w)
    return " ".join(out), changed


def parse_cell(raw: str):
    s = raw.replace("−", "-").replace(" ", " ").replace(" ", " ")
    s, digit_fixed = fix_digit_confusions(re.sub(r"\s+", " ", s).strip())
    if s.lower() in NA:
        return {"kind": "empty", "value": None, "unit": None, "flag": "ok"}
    if not re.search(r"[0-9A-Za-z€$%]", s):
        # only specks / punctuation in an otherwise blank cell, e.g. "." or "·:"
        return {"kind": "empty", "value": None, "unit": None, "flag": "noise_only"}
    s, unit = split_unit(s)
    unit_flag = None
    if unit is None:
        # OCR-damaged unit like "m5" or "rns": number first, then a short junk suffix
        m2 = re.fullmatch(r"([-(]?\d[\d.,: ]*\d\)?-?)\s+([A-Za-z0-9€$%]{1,4})", s)
        if m2 and re.search(r"[A-Za-z€$%]", m2.group(2)):
            s, unit, unit_flag = m2.group(1), m2.group(2), "unit_check"
    if re.search(r"[A-Za-z]", s):
        return {"kind": "text", "value": None, "text": raw.strip(), "unit": None, "flag": "ok"}

    neg, flag = False, "ok"
    s = s.strip()
    opens, closes = s.startswith("("), s.endswith(")")
    if opens or closes:
        neg, s = True, s.strip("()").strip()
        if opens != closes:
            flag = "check_sign"
    if s.endswith("-"):
        neg, s = True, s[:-1].strip()
    if s.startswith("-"):
        neg, s = True, s[1:].strip()

    m = NUM.fullmatch(s)
    if not m:
        return {"kind": "number", "value": "UNPARSED", "raw": raw, "unit": unit, "flag": "check"}
    int_part, sep, dec = m.groups()
    value = float(re.sub(r"\D", "", int_part) + ("." + dec if dec else ""))
    if not dec and re.fullmatch(r"\d{1,3}[.,]\d{3}", int_part):
        flag = "ambiguous_separator"   # "1.234": thousands or decimal? decide by table vote
    if unit is not None:
        unit = UNIT_FIXES.get(unit, unit)
        if unit not in KNOWN_UNITS:
            unit_flag = "unit_check"
    if unit_flag and flag == "ok":
        flag = unit_flag
    if digit_fixed and flag == "ok":
        flag = "digit_fixed"
    return {"kind": "number", "value": -value if neg else value, "unit": unit, "flag": flag,
            "decimals": len(dec) if dec else 0}


def repair_with_table_format(raw: str, table_decimals: int):
    """Fallback for fax noise: when a cell is UNPARSED but the table vote says every
    number has `table_decimals` decimals, ignore the (noisy) punctuation, keep only
    the digits and put the decimal point back by position."""
    s = raw.replace("−", "-")
    s, unit = split_unit(s.strip())
    neg = s.strip().startswith(("-", "(")) or s.strip().endswith((")", "-"))
    digits = re.sub(r"\D", "", s)
    if len(digits) <= table_decimals:
        return {"kind": "number", "value": "UNPARSED", "raw": raw, "unit": unit, "flag": "check"}
    value = int(digits) / (10 ** table_decimals)
    return {"kind": "number", "value": -value if neg else value, "unit": unit, "flag": "repaired"}


def table_decimals_vote(raw_cells, min_share: float = 0.8):
    """Decide how many decimals the numbers in THIS table have, using only cells
    that parsed cleanly (flag 'ok'). Returns None if there is no clear majority:
    then do NOT repair, just keep the cells flagged."""
    counts = {}
    for raw in raw_cells:
        r = parse_cell(raw)
        if r["kind"] == "number" and r["flag"] == "ok":
            counts[r["decimals"]] = counts.get(r["decimals"], 0) + 1
    if not counts:
        return None
    best, n = max(counts.items(), key=lambda kv: kv[1])
    return best if n / sum(counts.values()) >= min_share else None


CLEAN_FLAGS = ("ok", "ok_by_table", "digit_fixed")
_ALLOWED = re.compile(r"[\d.,() \-−]*")


def _number_part(raw):
    s = raw.replace("−", "-").replace(" ", " ").replace(" ", " ").strip()
    s, unit = split_unit(s)
    return s.strip(), unit


def _shape(raw, decimals):
    """(thousands_sep, decimal_sep) as printed in a cleanly parsed number.
    thousands_sep is None when the integer part is too short to tell."""
    s, _ = _number_part(raw)
    s = s.strip("()-").strip()
    dec_sep = None
    if decimals:
        dec_sep, s = s[-decimals - 1], s[:-decimals - 1]
    digits = re.sub(r"\D", "", s)
    if len(digits) < 4:
        return None, dec_sep
    seps = set(re.sub(r"\d", "", s))
    return (seps.pop() if len(seps) == 1 else ("" if not seps else "?")), dec_sep


def _vote(values, min_share):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, False
    best = max(set(vals), key=vals.count)
    return best, vals.count(best) / len(vals) >= min_share


def _rebuild(raw, decimals, thou, dec_sep):
    """Rebuild a number in the table's format from its digits alone."""
    s, unit = _number_part(raw)
    digits = re.sub(r"\D", "", s)
    if len(digits) <= decimals:
        return None
    int_d, dec_d = (digits[:-decimals], digits[-decimals:]) if decimals else (digits, "")
    int_d = int_d.lstrip("0") or "0"
    if thou:
        groups = []
        while len(int_d) > 3:
            groups.insert(0, int_d[-3:])
            int_d = int_d[:-3]
        int_d = thou.join([int_d] + groups)
    text = int_d + (dec_sep + dec_d if decimals else "")
    if "(" in s or ")" in s:
        text = f"({text})"
    elif s.startswith("-"):
        text = "-" + text
    elif s.endswith("-"):
        text = text + "-"
    if unit:
        text += " " + UNIT_FIXES.get(unit, unit)
    return text


def _needs_repair(raw, p, thou, dec_sep):
    if p["kind"] != "number":
        return False
    if p["flag"] not in CLEAN_FLAGS:
        return True                                  # format mismatch / unparsed / sign
    s, unit = _number_part(raw)
    if not _ALLOWED.fullmatch(s):
        return True                                  # junk characters (: ; ' ° ‚ ...)
    t, d = _shape(raw, p.get("decimals", 0))
    return (t is not None and t != thou) or (d is not None and d != dec_sep)


def resolve_table(raw_cells, min_share: float = 0.8, with_style: bool = False):
    """Parse all value cells of ONE table together.
    1. parse each cell
    2. vote how many decimals this table uses ('ok' cells, plus 'ambiguous_separator'
       cells, which look like integers with a thousands separator, e.g. '59,333')
    3. repair UNPARSED cells with the voted decimals
    4. if the table uses 0 or 2 decimals, '59,333' cannot be a 3-decimal number,
       so ambiguous cells are confirmed as thousands ('ok_by_table')"""
    parsed = [parse_cell(r) for r in raw_cells]
    counts = {}
    for p in parsed:
        if p["kind"] == "number" and p["flag"] in ("ok", "ambiguous_separator"):
            d = p.get("decimals", 0)
            counts[d] = counts.get(d, 0) + 1
    decimals = None
    if counts:
        best, n = max(counts.items(), key=lambda kv: kv[1])
        if n / sum(counts.values()) >= min_share:
            decimals = best
    out = []
    for raw, p in zip(raw_cells, parsed):
        if p["value"] == "UNPARSED" and decimals is not None:
            p = repair_with_table_format(raw, decimals)
        elif p["flag"] == "ambiguous_separator" and decimals in (0, 2):
            p = {**p, "flag": "ok_by_table"}
        elif (p["kind"] == "number" and p["flag"] == "ok" and decimals is not None
              and p.get("decimals", 0) != decimals):
            # e.g. "40.823.137" in a 2-decimal table: separators are noise (ch. 42)
            p = {**p, "flag": "format_mismatch"}
        out.append(p)

    # clear vote -> rebuild every number cell that breaks the table's format or
    # carries junk characters from its digits alone ('repaired', ch. 41-42)
    style = None
    if decimals is not None:
        shapes = [_shape(r, p.get("decimals", 0)) for r, p in zip(raw_cells, out)
                  if p["kind"] == "number" and p["flag"] in CLEAN_FLAGS
                  and p.get("decimals", 0) == decimals]
        # "42.074.03": same char as thousands and decimal separator = misread, no vote
        shapes = [s for s in shapes if s[0] is None or s[1] is None or s[0] != s[1]]
        thou, thou_ok = _vote([s[0] for s in shapes], min_share)
        dec_sep, dec_ok = _vote([s[1] for s in shapes], min_share) if decimals else (None, True)
        if thou_ok and dec_ok and thou != "?":
            style = {"decimals": decimals, "thou": thou, "dec_sep": dec_sep}
            for i, (raw, p) in enumerate(zip(raw_cells, out)):
                if not _needs_repair(raw, p, thou, dec_sep):
                    continue
                text = _rebuild(raw, decimals, thou, dec_sep)
                if text is None or text == raw.strip():
                    continue
                q = parse_cell(text)
                if q["kind"] == "number" and q["value"] != "UNPARSED":
                    out[i] = {**q, "flag": "repaired", "repaired_text": text,
                              "orig_flag": p["flag"]}
    return (out, decimals, style) if with_style else (out, decimals)


def _digits_sign(raw):
    s, unit = _number_part(raw)
    neg = s.startswith(("-", "(")) or s.endswith((")", "-"))
    return re.sub(r"\D", "", s), neg, unit


def vote_number(reads, style):
    """Digit-by-digit majority over several reads of one number cell (C: third
    reader). Every read is reduced to (digits, sign) -- separators and junk do
    not vote, the table format is applied afterwards. The first read is the
    tie-breaker. Returns the voted text in the table's format, or None."""
    parts = [_digits_sign(r) for r in reads if r]
    parts = [p for p in parts if p[0]]
    if len(parts) < 2 or style is None:
        return None
    d = style["decimals"]
    lengths = [len(p[0]) for p in parts]
    best_len = max(set(lengths), key=lambda L: (lengths.count(L), L == lengths[0]))
    if lengths.count(best_len) < 2 or best_len <= d:
        return None                                   # no two reads agree on the length
    same = [p for p in parts if len(p[0]) == best_len]
    digits = ""
    for i in range(best_len):
        col = [p[0][i] for p in same]
        digits += max(set(col), key=lambda ch: (col.count(ch), ch == col[0]))
    signs = [p[1] for p in parts]
    neg = max(set(signs), key=lambda v: (signs.count(v), v == signs[0]))
    unit = parts[0][2]
    paren = any("(" in r or ")" in r for r in reads if r)
    raw = (f"({digits})" if paren else f"-{digits}") if neg else digits
    return _rebuild(raw + (f" {unit}" if unit else ""), d, style["thou"], style["dec_sep"])
