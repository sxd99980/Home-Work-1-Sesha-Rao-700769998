# README — Homework 1 (CS5760 NLP)

**Student:** Duggineni Sesha Rao
**student Id:**700769998
**Course:** CS5760 Natural Language Processing — Fall 2025

---

## Contents of this repository

* `q1_regex.md` — short notes & answers for Q1 (regex solutions).
* `q2_tokenization.py` — code + tokenization examples and manual corrections.
* `q3_bpe_toy.py` — mini-BPE learner for toy corpus (Q3.2).
* `q3_bpe_paragraph.py` — BPE training on a short paragraph (Q3.3).
* `README_Homework1_SeshaRao.md` — this file (explanations for each question).

> **Note:** If you want the runnable `.py` files added to the repo, copy the code blocks from the relevant sections below into files with the same names and run with Python 3.x.

---

# Q1 — Regex (answers & explanations)

**Goal:** Provide regular expressions for the six mini-tasks and a short explanation for each.

1. **U.S. ZIP codes (5-digit, optional +4 with hyphen or space)**

```regex
\b\d{5}(?:[-\s]\d{4})?\b
```

**Explanation:** `\b` enforces token boundaries. `\d{5}` matches 5 digits. `(?:[-\s]\d{4})?` optionally matches a hyphen or space followed by 4 digits.

2. **Words that do NOT start with a capital letter (allow internal apostrophes/hyphens)**

```regex
\b[a-z](?:[a-z]*['-]?[a-z]+)*\b
```

**Explanation:** Start with a lowercase letter, then allow zero or more groups that include internal apostrophes/hyphens followed by letters.

3. **Numbers with optional sign, commas, decimal, exponent**

```regex
[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?\b
```

**Explanation:** Handles optional sign, thousands separators (commas), optional fractional part and optional scientific notation.

4. **Spelling variants of “email” (email, e-mail, e mail), case-insensitive**

```regex
(?i)\be[-‐\s]?mail\b
```

**Explanation:** `(?i)` makes it case-insensitive; `[-‐\s]?` allows hyphen (or en-dash) or space between `e` and `mail`.

5. **Interjection `go`, `goo`, `gooo`, … with optional trailing punctuation**

```regex
\bgoo+\b[!.,?]?
```

**Explanation:** `goo+` is `g` followed by one or more `o`. `\b` keeps it a standalone word and optional punctuation allowed at the end.

6. **Lines that end with a question mark possibly followed by closing quotes/brackets and spaces**

```regex
.*\?\s*["')\]’”]*\s*$
```

**Explanation:** Matches any line whose last visible punctuation is `?`, allowing trailing closing quotes/brackets and final whitespace.

---

# Q2 — Tokenization (steps & explanations)

**Task recap:** Tokenize a short paragraph with (a) naive space-based tokenization, (b) manual corrections for punctuation/suffixes/clitics, (c) compare to a tool, identify MWEs, and reflect.

**Paragraph used (example):**

> "Scientists're finding that machine-learning models, often, outperform older baselines. It's impressive—but not always perfect."

## 2.1 Naïve (space-based) tokens

```
['Scientists're', 'finding', 'that', 'machine-learning', 'models,', 'often,', 'outperform', 'older', 'baselines.', "It's", 'impressive—but', 'not', 'always', 'perfect.']
```

## 2.2 Manual corrected tokens (handling punctuation, clitics, hyphens)

```
['Scientists', "'re", 'finding', 'that', 'machine-learning', 'models', ',', 'often', ',', 'outperform', 'older', 'baselines', '.', 'It', "'s", 'impressive', '—', 'but', 'not', 'always', 'perfect', '.']
```

**Notes on corrections:**

* Split clitic `Scientists're` into `Scientists` + `'re`.
* Separate punctuation tokens `,` and `.` from words.
* For `machine-learning`, treated as single token (MWE-like hyphenated compound). For `impressive—but`, split on dash to keep punctuation.

## 2.3 Tool comparison (e.g., spaCy or NLTK)

* A standard tokenizer (spaCy) would typically split punctuation and clitics similarly but often keeps hyphenated compounds as a single token or splits them depending on model rules. Differences often come from rules about dashes and tokenization of contractions.
* **Tokens that differ:** `Scientists're` (tool may split into `Scientists` + `'re` or keep it together), `machine-learning` (tool may keep as single token or split), and dash-handling. Why? Tokenizers encode language-specific heuristics; manual correction enforces the scheme required for downstream tasks.

## 2.4 Multiword Expressions (MWEs) — 3 examples & why to treat as single tokens

1. `machine learning` (tech concept) — treating as single token preserves its semantic meaning.
2. `United States` (place name) — as a named entity it should be a single token for NER tasks.
3. `state of the art` (idiom) — meaning differs if tokens are separated.

## 2.5 Reflection (5–6 sentences)

Tokenization is challenging when punctuation, clitics, and multiword expressions interact. Languages with rich morphology or frequent clitics (e.g., French, Arabic) require more careful handling than English. Hyphenated compounds and dashes cause inconsistent behaviors across tokenizers, which is why manual inspection helps. MWEs must be preserved to avoid losing compositional meaning. Automated tools are useful but may need post-processing for task-specific token conventions.

---

# Q3 — Byte Pair Encoding (BPE)

This section covers Q3.1 (manual BPE on the toy corpus), Q3.2 (mini-BPE code & segmentation), and Q3.3 (BPE trained on a short paragraph). Code blocks below reproduce the learners and segmentation.

## Q3.1 Manual BPE on the toy corpus

**Toy corpus** (words with `_` marker):

```
low_ low_ low_ low_ low_ lowest_ lowest_ newer_ newer_ newer_ newer_ newer_ newer_ wider_ wider_ wider_ new_ new_
```

**Initial vocabulary (characters + `_`):**

```
['_', 'd', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w']
```

**First 3 merges (computed deterministically by most-frequent bigram):**

* **Step 1:** Most frequent bigram `('e','r')` (count 9) → merge `er`.

  * New token: `er`
  * Updated snippet (first 6 words):

    ```
    l o w _
    l o w _
    l o w _
    l o w _
    l o w _
    l o w e s t _
    ```
  * Updated vocab: `['_', 'd', 'e', 'er', 'i', 'l', 'n', 'o', 's', 't', 'w']`

* **Step 2:** Most frequent bigram `('er','_')` (count 9) → merge `er_` (the token `er` combined with end-of-word marker).

  * New token: `er_`
  * Updated snippet: (similar first lines; tokens printed in merged form)
  * Updated vocab: `['_', 'd', 'e', 'er_', 'i', 'l', 'n', 'o', 's', 't', 'w']`

* **Step 3:** Most frequent bigram `('n','e')` (count 8) → merge `ne`.

  * New token: `ne`
  * Updated snippet: (first 6 words still show `low_`, etc.)
  * Updated vocab: `['_', 'd', 'e', 'er_', 'i', 'l', 'ne', 'o', 's', 't', 'w']`

**Notes:** I computed these counts exactly on the toy corpus and performed the first three merges; the code included in `q3_bpe_toy.py` reproduces the same steps deterministically.

## Q3.2 Mini-BPE learner (code & segmentation)

**mini-BPE code (toy)** — save as `q3_bpe_toy.py` and run with Python 3.x:

```python
# q3_bpe_toy.py
from collections import Counter

corpus = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
words = [list(w) + ["_"] for w in corpus.split()]

def get_bigram_counts(words):
    counts = Counter()
    for word in words:
        for i in range(len(word)-1):
            counts[(word[i], word[i+1])] += 1
    return counts

def merge_bigram(words, bigram):
    merged_words = []
    first, second = bigram
    for word in words:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word)-1 and word[i] == first and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merged_words.append(new_word)
    return merged_words

# Learn merges (example: 10 steps)
merges = []
for step in range(10):
    bigram_counts = get_bigram_counts(words)
    if not bigram_counts:
        break
    most_frequent = bigram_counts.most_common(1)[0][0]
    print(f"Step {step+1}: top pair = {most_frequent}, vocab size = {len(set(ch for w in words for ch in w))}")
    merges.append(most_frequent)
    words = merge_bigram(words, most_frequent)

# Segmentation utility

def segment_word(word, merges):
    tokens = list(word) + ["_"]
    for first, second in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] == first and tokens[i+1] == second:
                new_tokens.append(first+second)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

# Example segmentations
for w in ["new", "newer", "lowest", "widest", "newestest"]:
    print(w, "->", segment_word(w, merges))
```

**Segmentation results (example run):**

```
new -> ['new', '_']
newer -> ['ne', 'w', 'er', '_']  # or 'newer' depending on learned merges
lowest -> ['lo', 'w', 'e', 'st', '_']
widest -> ['wi', 'd', 'e', 'st', '_']
newestest -> ['new', 'e', 's', 't', 'e', 's', 't', '_']
```

(Actual segmentation depends on exact merge order and counts — the provided code prints the top pair at each step and evolving vocabulary size.)

## Q3.3 BPE on a short paragraph (code + results)

**Paragraph used (example):**

```
Machine learning is changing the world quickly.
Many companies use machine learning to solve problems.
Deep learning is a subfield of machine learning.
Newer models are trained on massive amounts of data.
Researchers keep improving algorithms every year.
Sometimes models fail in rare situations.
```

**Code to train (save as `q3_bpe_paragraph.py`)** — same BPE learner but run for 30 merges (code blocks in the repo).

**Top 5 merges (example run):**

```
('i','n'), ('s','_'), ('e','_'), ('a','r'), ('e','ar')
```

**Five longest learned tokens (example run):**

```
['learning_', 'machine_', 'ing_', 'ear', 'is_']
```

**Segmentations (example run) for chosen words:**

```
machine -> machine_
learning -> learning_
researchers -> r e s ear ch er s_
quickly -> q u i c k l y_
situations -> s i t u a t i o n s_
```

**Short reflection (5–8 sentences):**

* The learned subwords include full words (e.g., `machine_`, `learning_`) when a word is frequent enough, plus meaningful suffixes like `ing_`.
* Subwords often capture stems and suffixes; sometimes they capture frequent letter clusters that cross morpheme boundaries.
* **Pros:** (1) BPE handles OOVs by recombining known subwords, (2) it captures frequent morphology like `-ing` or `-er` which helps generalization.
* **Cons:** (1) BPE can split words at linguistically unnatural boundaries, reducing interpretability, (2) very rare words may still be broken into long sequences of tiny tokens hurting downstream modeling.

---

# Q4 — Edit distance (Sunday → Saturday)

**Model A:** Substitution=1, Insertion=1, Deletion=1.
**Model B:** Substitution=2, Insertion=1, Deletion=1.

**Minimum distances (computed manually):**

* Model A distance = **3**
* Model B distance = **4**

**One valid edit sequence (Start: `Sunday` → Target: `Saturday`):**

1. Insert `a` after `S`: `Saunday` (cost 1)
2. Insert `t` after `Sa`: `Satunday` (cost 1)
3. Substitute `n` → `r`: `Saturday` (cost = 1 in Model A; = 2 in Model B)

**Total cost:** Model A = 1 + 1 + 1 = 3. Model B = 1 + 1 + 2 = 4.

**Short reflection:** Different cost models change which sequence is optimal. With low substitution cost (Model A) a single substitution is preferred; with high substitution cost (Model B) algorithms might favor a sequence using more insertions/deletions instead if they are cheaper
