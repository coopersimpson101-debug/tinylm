"""Document cleaning: Unicode normalisation plus language / quality heuristics.

``normalize`` applies NFKC (folds compatibility characters: NBSP -> space,
"..." -> "...", full-width forms -> ASCII), then folds typographic quotes and
dashes to ASCII, strips control / zero-width characters, and normalises
whitespace (CRLF -> LF, trailing spaces, runs of blank lines).

``quality_check`` returns ``None`` for a document that passes, or the name of
the first heuristic it fails. The thresholds follow the spirit of the Gopher /
C4 rules (Rae et al. 2021; Raffel et al. 2020), tuned for short English prose:

* length:       >= ``min_chars`` characters and >= ``min_words`` words
* alpha:        >= 70 % of non-space characters are letters
* word length:  mean word length in [2.5, 10]
* english:      >= 15 % of words are common English function words, and
                <= 1 % of characters are non-ASCII after normalisation
* repetition:   <= 30 % duplicate lines, and the most frequent word 3-gram
                covers <= 20 % of all 3-grams
* markup:       no leftover separator / template tokens (``<|``)
"""
import re
import unicodedata
from collections import Counter

_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u00b4": "'",
})
_ZERO_WIDTH = re.compile("[\u200b-\u200d\u2060\ufeff\u00ad]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SPACES = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")

STOPWORDS = frozenset("""a an the and or but if then so of to in on at by for with from as is are was
were be been am it its he she they we you i his her their our your my me him them us this that
these those there here not no do did does had has have will would can could should said""".split())


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_FOLD)
    t = _ZERO_WIDTH.sub("", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _CTRL.sub("", t)
    t = "\n".join(_SPACES.sub(" ", line).strip() for line in t.split("\n"))
    t = _BLANKS.sub("\n\n", t)
    return t.strip()


def quality_check(text: str, min_chars: int = 100, min_words: int = 20,
                  max_chars: int = 100_000) -> str | None:
    if "<|" in text:
        return "markup"
    n = len(text)
    if n < min_chars or n > max_chars:
        return "length"
    words = text.split()
    if len(words) < min_words:
        return "length"
    nonspace = n - text.count(" ") - text.count("\n")
    alpha = sum(ch.isalpha() for ch in text)
    if alpha < 0.70 * max(nonspace, 1):
        return "alpha"
    mean_len = sum(len(w) for w in words) / len(words)
    if not 2.5 <= mean_len <= 10:
        return "word_length"
    lw = [w.strip(".,!?;:\"'()").lower() for w in words]
    if sum(w in STOPWORDS for w in lw) < 0.15 * len(lw):
        return "english"
    if not text.isascii() and sum(ord(ch) > 127 for ch in text) > 0.01 * n:
        return "english"
    lines = [line for line in text.split("\n") if line]
    if len(lines) >= 4 and 1 - len(set(lines)) / len(lines) > 0.30:
        return "repetition"
    if len(lw) >= 30:
        tri = Counter(zip(lw, lw[1:], lw[2:]))
        if tri.most_common(1)[0][1] > 0.20 * (len(lw) - 2):
            return "repetition"
    return None


def split_documents(path: str, sep: str = "<|endoftext|>", truncated: bool = False,
                    chunk: int = 32 << 20):
    """Yield documents from a separator-delimited text file, streaming.

    With ``truncated=True`` (the file is a byte-range prefix of a larger file)
    the trailing partial document is discarded.
    """
    buf = ""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            buf += b
            parts = buf.split(sep)
            buf = parts.pop()
            yield from parts
    if buf.strip() and not truncated:
        yield buf
