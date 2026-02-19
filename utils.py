from dataclasses import dataclass
from enum import Enum
from typing import Optional, Iterable, List, Callable, Literal, Optional, Dict, Tuple
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import spacy
import re
from itertools import islice

class PCLBinaryLabel(Enum):
    NO_PCL = 0
    PCL = 1


@dataclass(frozen=True)
class PCLItem:
    par_id: int
    art_id: int
    keyword: str          # raw string
    country_code: str     # ISO alpha-2
    text: str
    label_ordinal: int    # 0–4
    label_binary: PCLBinaryLabel

    @property
    def is_pcl(self) -> bool:
        return self.label_binary == PCLBinaryLabel.PCL


def parse_label_binary(label: int) -> PCLBinaryLabel:
    return PCLBinaryLabel.PCL if label >= 2 else PCLBinaryLabel.NO_PCL


def parse_item(line: str) -> PCLItem:
    # split max 5 times so text can contain tabs safely
    parts = line.rstrip("\n").split("\t", maxsplit=5)

    if len(parts) != 6:
        raise ValueError(f"Invalid line format ({len(parts)} fields): {line}")

    par_id_str, art_id_str, keyword, country_code, text, label_str = parts

    par_id = int(par_id_str)
    art_id = int(art_id_str[2:]) # drop @s
    label = int(label_str)

    if not (0 <= label <= 4):
        raise ValueError(f"Invalid label {label} in line: {line}")

    return PCLItem(
        par_id=par_id,
        art_id=art_id,
        keyword=keyword.lower(),
        country_code=country_code.upper(),
        text=text,
        label_ordinal=label,
        label_binary=parse_label_binary(label),
    )

from typing import Iterable


def iter_dataset(path: str) -> Iterable[PCLItem]:
    with open(path, "r", encoding="utf-8") as f:
        # skip first 5 lines (metadata / headers)
        for _ in range(5):
            next(f, None)

        for line_num, line in enumerate(f, start=6):
            if not line.strip():
                continue
            try:
                yield parse_item(line)
            except Exception as e:
                raise RuntimeError(f"Error parsing line {line_num}") from e

def load_dataset(path: str):
    return list(iter_dataset(path))

def label_stats(items: Iterable[PCLItem]):
    ordinal = Counter(x.label_ordinal for x in items)
    binary = Counter(x.label_binary for x in items)
    return ordinal, binary


LabelMode = Literal["binary", "ordinal"]


class PCLDataset(Dataset):
    def __init__(
        self,
        items: List[PCLItem],
        tokenizer: Optional[Callable] = None,
        max_length: int = 256,
        label_mode: LabelMode = "binary",
    ):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mode = label_mode

        if label_mode not in {"binary", "ordinal"}:
            raise ValueError(f"Invalid label_mode: {label_mode}")

    def __len__(self):
        return len(self.items)

    def _get_label(self, item: PCLItem) -> int:
        if self.label_mode == "binary":
            return item.label_binary.value
        else:  # ordinal
            return item.label_ordinal

    def __getitem__(self, idx):
        item = self.items[idx]
        text = item.text
        label = self._get_label(item)

        if self.tokenizer is None:
            return {
                "text": text,
                "label": torch.tensor(label, dtype=torch.long),
                "meta": {
                    "par_id": item.par_id,
                    "keyword": item.keyword,
                    "country": item.country_code,
                },
            }

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        
# Tokenisers

def whitespace_tokenise(text: str):
    return text.split()


TOKEN_RE = re.compile(r"\b\w+\b")

def regex_tokenise(text: str):
    return TOKEN_RE.findall(text.lower())

nlp = spacy.load("en_core_web_sm")
def spacy_tokenise(text: str):
    return [tok.text for tok in nlp(text)]

bpe = AutoTokenizer.from_pretrained("gpt2")
def bpe_tokenise(text: str):
    return bpe.tokenize(text)

def char_tokenise(text: str):
    return list(text)

Tokeniser = Callable[[str], List[str]]

TOKENISERS = {
    "whitespace": whitespace_tokenise,
    "regex": regex_tokenise,
    "spacy": spacy_tokenise,
    "bpe": bpe_tokenise,
    "char": char_tokenise,
}

# EDA

# Basic Stats
# use regex and bpe
def corpus_stats(items, tokenizer):
    lengths = [len(tokenizer(x.text)) for x in items]
    
    return {
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": sum(lengths) / len(lengths),
        "vocab_size": len({tok for x in items for tok in tokenizer(x.text)}),
    }

# Lexical Analysis (word level)
# use regex
def ngrams(tokens, n):
    return zip(*(tokens[i:] for i in range(n)))

def top_ngrams(items, tokenizer, n=2, k=20):
    counter = Counter()
    for x in items:
        tokens = tokenizer(x.text)
        counter.update(ngrams(tokens, n))
    return counter.most_common(k)

def stop_word_density(items):
    total_tokens = 0
    stop_tokens = 0
    for x in items:
        text = x.text
        tokens = nlp(text)
        total_tokens += len(tokens)
        stop_tokens += len([t for t in tokens if t.is_stop])
    return stop_tokens / total_tokens

def top_words(items, k=20):
    counter = Counter()
    for x in items:
        tokens = nlp(x.text)
        filtered = [t.lemma_.lower() for t in tokens if not t.is_stop and t.is_alpha]
        counter.update(filtered)
    return counter.most_common(k)
        

# Semantic & Synctatic Exploration
# use spacy
def pos_distribution(items):
    from collections import Counter
    counter = Counter()
    for x in items:
        doc = nlp(x.text)
        counter.update(tok.pos_ for tok in doc)
    return counter

# Noise and Artifacts
# use char and regex
def find_weird_chars(items):
    weird = Counter()
    for x in items:
        for c in x.text:
            if not c.isprintable():
                weird[c] += 1
    return weird


# --- Enhanced EDA utilities ---

def length_distribution(items, tokenizer):
    """Return token lengths and key percentiles for max_length selection."""
    lengths = np.array([len(tokenizer(x.text)) for x in items])
    percentiles = {f"p{p}": int(np.percentile(lengths, p)) for p in [50, 75, 90, 95, 99]}
    return {
        "lengths": lengths,
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": int(np.median(lengths)),
        "std": float(lengths.std()),
        **percentiles,
    }


def class_top_words(items, k=20):
    """Top-k words (stop-words removed) split by PCL vs non-PCL."""
    pcl_counter = Counter()
    no_pcl_counter = Counter()
    for x in items:
        tokens = nlp(x.text)
        filtered = [t.lemma_.lower() for t in tokens if not t.is_stop and t.is_alpha]
        if x.is_pcl:
            pcl_counter.update(filtered)
        else:
            no_pcl_counter.update(filtered)
    return {"pcl": pcl_counter.most_common(k), "no_pcl": no_pcl_counter.most_common(k)}


def find_duplicates(items):
    """Find duplicate texts. Returns list of (text, count) for count > 1."""
    text_counts = Counter(x.text for x in items)
    dupes = [(t, c) for t, c in text_counts.most_common() if c > 1]
    return dupes


def find_label_disagreements(items):
    """Texts that appear multiple times with different labels."""
    from collections import defaultdict
    text_labels = defaultdict(set)
    for x in items:
        text_labels[x.text].add(x.label_ordinal)
    return {t: labels for t, labels in text_labels.items() if len(labels) > 1}


def find_outliers(items, tokenizer, low_pct=1, high_pct=99):
    """Flag sequences shorter/longer than the given percentiles."""
    lengths = [(x, len(tokenizer(x.text))) for x in items]
    all_lens = np.array([l for _, l in lengths])
    lo = np.percentile(all_lens, low_pct)
    hi = np.percentile(all_lens, high_pct)
    short = [(x, l) for x, l in lengths if l < lo]
    long_ = [(x, l) for x, l in lengths if l > hi]
    return {"short": short, "long": long_, "lo_thresh": lo, "hi_thresh": hi}


def ner_distribution(items):
    """Distribution of named-entity types across the corpus."""
    counter = Counter()
    examples = {}
    for x in items:
        doc = nlp(x.text)
        for ent in doc.ents:
            counter[ent.label_] += 1
            if ent.label_ not in examples:
                examples[ent.label_] = ent.text
    return counter, examples


def keyword_label_correlation(items):
    """Check whether the keyword field alone could predict PCL.
    Returns {keyword: {pcl_count, total_count, pcl_rate}}."""
    from collections import defaultdict
    kw_stats = defaultdict(lambda: {"pcl": 0, "total": 0})
    for x in items:
        kw_stats[x.keyword]["total"] += 1
        if x.is_pcl:
            kw_stats[x.keyword]["pcl"] += 1
    return {
        kw: {**v, "pcl_rate": v["pcl"] / v["total"] if v["total"] else 0}
        for kw, v in sorted(kw_stats.items(), key=lambda t: -t[1]["total"])
    }


def html_and_entity_scan(items):
    """Detect HTML tags, entities, URLs, and other artefacts."""
    patterns = {
        "html_tags": re.compile(r"<[^>]+>"),
        "html_entities": re.compile(r"&[a-z]+;"),
        "urls": re.compile(r"https?://\S+"),
        "escaped_newlines": re.compile(r"\\n"),
    }
    results = {name: [] for name in patterns}
    for x in items:
        for name, pat in patterns.items():
            matches = pat.findall(x.text)
            if matches:
                results[name].append((x.par_id, matches))
    return {name: hits for name, hits in results.items()}


def avg_sentence_length(items, sample=500):
    """Average number of sentences per document (via spaCy sentencizer)."""
    import random
    subset = random.sample(items, min(sample, len(items)))
    sent_counts = [len(list(nlp(x.text).sents)) for x in subset]
    return {"mean_sents": np.mean(sent_counts), "max_sents": max(sent_counts), "min_sents": min(sent_counts)}


# --- Preprocessing pipeline ---

import html as _html

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_SMART_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'",   # single curly quotes
    "\u201c": '"', "\u201d": '"',   # double curly quotes
    "\u2013": "-", "\u2014": "-",   # en/em dash
    "\u2026": "...",                 # ellipsis
})


def clean_text(text: str) -> str:
    """Core text cleaning: HTML strip, entity decode, normalise whitespace/quotes."""
    text = _HTML_TAG_RE.sub("", text)           # strip HTML tags (<h> etc.)
    text = _html.unescape(text)                 # decode &amp; &gt; etc.
    text = text.translate(_SMART_QUOTES)         # normalise smart quotes/dashes
    text = _MULTI_SPACE_RE.sub(" ", text)        # collapse multiple spaces
    text = text.strip()
    return text


def mask_entities(text: str, nlp_model=None) -> str:
    """Replace named entities with their type tokens, e.g. 'Libya' → '[GPE]'.
    Uses the module-level `nlp` if no model is provided."""
    model = nlp_model or nlp
    doc = model(text)
    # Process spans in reverse order so character offsets stay valid
    chars = list(text)
    for ent in reversed(doc.ents):
        replacement = f"[{ent.label_}]"
        chars[ent.start_char:ent.end_char] = list(replacement)
    return "".join(chars)


def preprocess_items(
    items: List[PCLItem],
    min_tokens: int = 3,
    do_entity_mask: bool = False,
    tokenizer: Callable = None,
) -> List[PCLItem]:
    """Apply the full preprocessing pipeline to a list of PCLItems.
    
    Steps:
      1. clean_text (HTML, entities, whitespace, quotes)
      2. Optionally mask named entities
      3. Filter out empty / near-empty samples (< min_tokens)
    
    Returns new PCLItem objects with cleaned text (frozen dataclass,
    so we create replacements).
    """
    if tokenizer is None:
        tokenizer = regex_tokenise

    cleaned = []
    dropped = {"empty": 0, "short": 0}

    for item in items:
        text = clean_text(item.text)

        if not text:
            dropped["empty"] += 1
            continue

        if len(tokenizer(text)) < min_tokens:
            dropped["short"] += 1
            continue

        if do_entity_mask:
            text = mask_entities(text)

        cleaned.append(PCLItem(
            par_id=item.par_id,
            art_id=item.art_id,
            keyword=item.keyword,
            country_code=item.country_code,
            text=text,
            label_ordinal=item.label_ordinal,
            label_binary=item.label_binary,
        ))

    print(f"Preprocessing: {len(items)} → {len(cleaned)} samples "
          f"(dropped {dropped['empty']} empty, {dropped['short']} short [<{min_tokens} tokens])")
    return cleaned


def keyword_stratified_split(
    items: List[PCLItem],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[PCLItem]]:
    """Keyword-stratified train/val/test split.
    
    Stratifies on (keyword, label_binary) to ensure each split has
    the same keyword distribution and class balance.
    """
    from sklearn.model_selection import train_test_split

    # Create stratification key: keyword + binary label
    strat_keys = [f"{item.keyword}_{item.label_binary.value}" for item in items]

    # First split: train+val vs test
    train_val, test, strat_tv, _ = train_test_split(
        items, strat_keys, test_size=test_size, random_state=seed, stratify=strat_keys,
    )

    # Second split: train vs val (relative to train+val size)
    relative_val = val_size / (1 - test_size)
    strat_tv_keys = [f"{item.keyword}_{item.label_binary.value}" for item in train_val]
    train, val, _, _ = train_test_split(
        train_val, strat_tv_keys, test_size=relative_val, random_state=seed, stratify=strat_tv_keys,
    )

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        pcl_count = sum(1 for x in split_data if x.is_pcl)
        print(f"  {split_name}: {pcl_count}/{len(split_data)} PCL ({pcl_count/len(split_data)*100:.1f}%)")

    return {"train": train, "val": val, "test": test}


def load_official_splits(
    items: List[PCLItem],
    splits_dir: str = "Dont_Patronize_Me_Trainingset/practice splits",
) -> Dict[str, List[PCLItem]]:
    """Split items according to the official SemEval train/dev par_id lists.

    The split CSV files contain ``par_id,label`` rows.  We only use par_id
    to partition *items* into ``train`` and ``dev`` sets.  Items whose
    par_id appears in neither file are reported but excluded.

    Returns
    -------
    dict with keys ``"train"`` and ``"dev"``, each a list of PCLItem.
    """
    import csv, os

    def _load_par_ids(path: str) -> set:
        ids = set()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.add(int(row["par_id"]))
        return ids

    train_ids = _load_par_ids(os.path.join(splits_dir, "train_semeval_parids-labels.csv"))
    dev_ids   = _load_par_ids(os.path.join(splits_dir, "dev_semeval_parids-labels.csv"))

    train, dev, unmatched = [], [], []
    for item in items:
        if item.par_id in train_ids:
            train.append(item)
        elif item.par_id in dev_ids:
            dev.append(item)
        else:
            unmatched.append(item)

    print(f"Official split: train={len(train)}, dev={len(dev)}")
    if unmatched:
        print(f"  (!) {len(unmatched)} items not in either split file — excluded")
    for name, data in [("train", train), ("dev", dev)]:
        pcl = sum(1 for x in data if x.is_pcl)
        print(f"  {name}: {pcl}/{len(data)} PCL ({pcl/len(data)*100:.1f}%)")

    return {"train": train, "dev": dev}

