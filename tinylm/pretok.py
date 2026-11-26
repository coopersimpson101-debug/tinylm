"""Qwen2 / GPT-4-style pre-tokeniser, written by hand.

The reference pattern (used by Qwen2 and, modulo the digit rule, by GPT-4's
cl100k) is

    (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\\r\\n\\p{L}\\p{N}]?\\p{L}+ | \\p{N}
    | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]* | \\s*[\\r\\n]+ | \\s+(?!\\S) | \\s+

``pretokenize`` is a direct matcher for that pattern: at each position it tries
the alternatives in order, exactly as a backtracking regex engine would, using
Unicode general categories from ``unicodedata`` (L* = letter, N* = number) and
the Unicode White_Space set. No regex engine is involved.

``pretokenize_fast`` is an optional speed path for pure-ASCII text that uses
the standard-library ``re`` module with the ASCII specialisation of the same
pattern; ``tests/test_pretok.py`` checks it agrees with the hand-written
matcher, and falls back to it for any non-ASCII input.
"""
import re
import unicodedata

# Unicode White_Space property (matches \s in the `regex` module and tinyinfer's C matcher)
_WS = frozenset([0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F,
                 0x205F, 0x3000] + list(range(0x2000, 0x200B)))
_cat_cache: dict = {}


def _cls(ch: str) -> int:
    """0 = other, 1 = letter, 2 = number, 3 = space."""
    c = _cat_cache.get(ch)
    if c is None:
        o = ord(ch)
        if o in _WS:
            c = 3
        else:
            cat = unicodedata.category(ch)
            c = 1 if cat[0] == "L" else 2 if cat[0] == "N" else 0
        _cat_cache[ch] = c
    return c


def pretokenize(text: str) -> list[str]:
    n = len(text)
    cls = [_cls(ch) for ch in text]
    out = []
    i = 0
    while i < n:
        c = cls[i]
        ch = text[i]
        j = -1
        # 1. contractions (case-insensitive)
        if ch == "'" and i + 1 < n:
            c1 = text[i + 1].lower()
            c2 = text[i + 2].lower() if i + 2 < n else ""
            if c1 in "stmd":
                j = i + 2
            elif c1 + c2 in ("re", "ve", "ll"):
                j = i + 3
        # 2. optional non-letter/number/newline prefix + letters
        if j < 0:
            k = -1
            if c == 1:
                k = i
            elif ch not in "\r\n" and c != 2 and i + 1 < n and cls[i + 1] == 1:
                k = i + 1
            if k >= 0:
                while k < n and cls[k] == 1:
                    k += 1
                j = k
        # 3. a single number character
        if j < 0 and c == 2:
            j = i + 1
        # 4. optional space + run of "other" + trailing newlines
        if j < 0:
            k = i + 1 if (ch == " " and i + 1 < n and cls[i + 1] == 0) else i
            if k < n and cls[k] == 0:
                while k < n and cls[k] == 0:
                    k += 1
                while k < n and text[k] in "\r\n":
                    k += 1
                j = k
        # 5-7. whitespace
        if j < 0 and c == 3:
            e = i
            last_nl = -1
            while e < n and cls[e] == 3:
                if text[e] in "\r\n":
                    last_nl = e
                e += 1
            if last_nl >= 0:          # \s*[\r\n]+
                j = last_nl + 1
            elif e == n:              # \s+(?!\S) at end of text
                j = e
            elif e - i >= 2:          # \s+(?!\S): leave one space for the next word
                j = e - 1
            else:                     # \s+
                j = e
        if j < 0:
            j = i + 1
        out.append(text[i:j])
        i = j
    return out


_ASCII_PAT = re.compile(
    r"'(?:[sdmtSDMT]|[rR][eE]|[vV][eE]|[lL][lL])|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]"
    r"| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", re.ASCII)


def pretokenize_fast(text: str) -> list[str]:
    """Same result as ``pretokenize``; uses ``re`` for pure-ASCII text."""
    if text.isascii():
        return _ASCII_PAT.findall(text)
    return pretokenize(text)
