# Word-Level Tokenization

Tokenization is the process of converting raw text into a sequence of discrete integer IDs that a neural network can process. It is the very first stage of every NLP pipeline, bridging the gap between human-readable strings and the numerical tensors that models operate on. Word-level tokenization, the simplest strategy, treats each whitespace-delimited word as an atomic token.

---

## What It Is

A word-level tokenizer performs two complementary operations. **Encoding** takes a raw string, normalizes it by lowercasing, splits it on whitespace, and maps each resulting word to a unique integer ID via a fixed vocabulary lookup. **Decoding** reverses the process, converting a sequence of integer IDs back into a human-readable string by looking up each ID in the reverse mapping and joining the words with spaces.

The vocabulary is built in advance from a training corpus. Every unique word receives a permanent integer ID. Words encountered at inference time that were never seen during training are mapped to a special unknown token. This is the core tradeoff: simplicity at the cost of a rigid vocabulary that cannot handle novel words. The tokenizer maintains a deterministic bijective mapping:

$$
\texttt{word\_to\_id}: \text{string} \rightarrow \text{integer} \qquad \texttt{id\_to\_word}: \text{integer} \rightarrow \text{string}
$$

---

## Key Equations

The encoding function for a single word $w$, given vocabulary $V$, is:

$$
\text{encode}(w) = \begin{cases} \texttt{word\_to\_id}[w] & \text{if } w \in V \\ 1 & \text{otherwise (UNK)} \end{cases}
$$

The four special tokens occupy the first four IDs with fixed assignments:

$$
\texttt{} = 0, \quad \texttt{} = 1, \quad \texttt{} = 2, \quad \texttt{} = 3
$$

Regular vocabulary words receive IDs starting from $4$. If the unique words from the training corpus, sorted alphabetically, are $w_0, w_1, \ldots, w_{n-1}$, then:

$$
\texttt{word\_to\_id}[w_i] = i + 4 \quad \text{for } i = 0, 1, \ldots, n-1
$$

The total vocabulary size is $|V| = n + 4$, where $n$ is the count of unique training words and $4$ accounts for special tokens.

The decoding function for a single ID $k$ is simply the inverse lookup:

$$
\text{decode}(k) = \texttt{id\_to\_word}[k]
$$

For a full sentence, encoding maps the lowercased, whitespace-split word list through $\text{encode}$ element-wise, and decoding maps IDs through $\text{decode}$ then joins with spaces.

---

## Vocabulary Building

Building the vocabulary is a deterministic, multi-step procedure that must be followed in exact order to produce consistent ID assignments.

**Step 1 - Initialize special tokens.** Reserve the first four IDs: `` = 0, `` = 1, `` = 2, `` = 3. These are hardcoded constants, never derived from training data.

**Step 2 - Lowercase all text.** Every training sentence is converted to lowercase so that "The", "the", and "THE" all map to the same token. Without this, the vocabulary contains spurious duplicates differing only in capitalization.

**Step 3 - Split on whitespace.** Each lowercased sentence is split into words using whitespace as the delimiter. Punctuation attached to a word (like "hello,") remains part of that token.

**Step 4 - Collect unique words.** Iterate through every word across all training sentences and collect the set of unique words. Duplicates are eliminated.

**Step 5 - Sort alphabetically.** The unique words are sorted in lexicographic order. Sorting is critical for reproducibility; without it, hash randomization or insertion order could produce different ID assignments across runs.

**Step 6 - Assign IDs.** Walk through the sorted list and assign IDs starting from $4$. The first word alphabetically gets ID $4$, the second $5$, and so on.

The result is a forward dictionary `word_to_id` and a reverse dictionary `id_to_word`, both including the special tokens.

---

## Special Tokens

Special tokens serve structural roles that regular words cannot. Each exists for a specific engineering reason rooted in how neural networks process sequences.

### PAD (ID = 0)

Neural networks process batches of sequences simultaneously, but sequences rarely have the same length. **Padding** fills shorter sequences so all can be stacked into a single tensor. PAD is assigned ID $0$ by convention, which interacts cleanly with masking: a binary mask of non-padding positions is simply a check for non-zero IDs. During attention, padding positions are masked out.

### UNK (ID = 1)

The **unknown token** is a fallback for any word not in the vocabulary. Every out-of-vocabulary (OOV) word collapses to the same UNK ID, destroying all information about the original word. This information loss is the central weakness of word-level tokenization.

### BOS (ID = 2)

The **beginning-of-sequence** token marks where a sequence starts. In autoregressive models, BOS serves as the initial input token the model conditions on to generate the first real word. Without it, the model cannot distinguish "the first token of a new sequence" from a continuation token.

### EOS (ID = 3)

The **end-of-sequence** token marks where a sequence terminates. In autoregressive generation, predicting EOS tells the decoding algorithm to stop. In the original Transformer, EOS on the source side signals input completion, and EOS on the target side signals the end of translation.

---

## Encoding: Text to IDs

Encoding converts a raw text string into a list of integer IDs in three steps.

**Step 1 - Lowercase.** Convert the entire input to lowercase, matching the normalization applied during vocabulary building.

**Step 2 - Split.** Split the lowercased string on whitespace to produce a list of word strings.

**Step 3 - Lookup.** For each word, look it up in `word_to_id`. If found, use its ID. If not found, use the UNK ID ($1$).

The output is a list of integers with the same length as the number of words. BOS and EOS are not automatically added; whether to wrap the sequence with them depends on the downstream model.

---

## Decoding: IDs to Text

Decoding converts a list of integer IDs back into a human-readable string.

**Step 1 - Lookup.** For each ID, look it up in `id_to_word` to retrieve the corresponding string. Special token IDs map to their string representations (``, ``, ``, ``).

**Step 2 - Join.** Concatenate all retrieved strings with a single space between each pair.

Decoding is not a perfect inverse of encoding. Any word that was mapped to UNK during encoding decodes as the literal string ``, not the original word. The original word is irrecoverably lost.

---

## Paper Context

The original Transformer paper, "Attention Is All You Need" (Vaswani et al., 2017), did not use word-level tokenization. The authors employed **Byte Pair Encoding (BPE)** with a shared source-target vocabulary of approximately 37,000 tokens for English-German translation and 32,000 tokens for English-French.

The choice was deliberate. Machine translation handles two languages simultaneously, and word-level vocabularies for both combined would be enormous. BPE compresses the vocabulary by representing rare words as sequences of common subword units, eliminating the OOV problem entirely.

Notably, Vaswani et al. used a shared vocabulary: a single BPE vocabulary learned from concatenated bilingual data. This allowed encoder and decoder to share the same embedding matrix, reducing parameters and exploiting cross-lingual subword overlap.

Word-level tokenization remains foundational as the conceptual baseline from which all subword methods depart. Every subword tokenizer still produces integer ID sequences, manages special tokens, and maintains encode/decode symmetry.

---

## Numerical Example

Consider two training sentences:

* **Sentence 1:** "The cat sat on the mat"
* **Sentence 2:** "The dog sat on the log"

### Building the Vocabulary

**Lowercase:** Both sentences become "the cat sat on the mat" and "the dog sat on the log".

**Collect unique words:** {cat, dog, log, mat, on, sat, the}. "the" appears four times but counts once.

**Sort alphabetically:** [cat, dog, log, mat, on, sat, the].

**Assign IDs starting from 4:**

* **``** = 0
* **``** = 1
* **``** = 2
* **``** = 3
* **cat** = 4
* **dog** = 5
* **log** = 6
* **mat** = 7
* **on** = 8
* **sat** = 9
* **the** = 10

Total vocabulary size: $7 + 4 = 11$.

### Encoding a Known Sentence

**Input:** "The cat sat on the mat"

**Lowercase:** "the cat sat on the mat"

**Split:** ["the", "cat", "sat", "on", "the", "mat"]

**Lookup each word:** [10, 4, 9, 8, 10, 7]

Every word exists in the vocabulary, so no UNK tokens appear.

### Encoding a Sentence With Unknown Words

**Input:** "The bird sat on the log"

**Lowercase:** "the bird sat on the log"

**Split:** ["the", "bird", "sat", "on", "the", "log"]

**Lookup:** "the" = 10, "bird" not in vocabulary = 1 (UNK), "sat" = 9, "on" = 8, "the" = 10, "log" = 6.

**Result:** [10, 1, 9, 8, 10, 6]

"bird" was never seen during training, so it collapses to UNK.

### Decoding

**Input IDs:** [10, 1, 9, 8, 10, 6]

**Lookup:** 10 = "the", 1 = "``", 9 = "sat", 8 = "on", 10 = "the", 6 = "log".

**Join with spaces:** "the `` sat on the log"

"bird" has been replaced by ``, demonstrating the irreversible information loss.

---

## Word-Level vs Subword Tokenization

Word-level and subword tokenization make fundamentally different tradeoffs along the granularity spectrum.

### Word-Level Tokenization

* **Vocabulary size:** Grows with the corpus. English has over 170,000 words in active use. A 100,000+ token vocabulary means the embedding matrix alone consumes hundreds of megabytes.
* **OOV problem:** Any word not in the vocabulary maps to UNK. Catastrophic for morphologically rich languages (Turkish, Finnish), proper nouns, and typos.
* **Token semantics:** Each token is a complete word, making the vocabulary human-interpretable.
* **Sequence length:** Produces the shortest sequences (one token per word), minimizing the $O(n^2)$ attention cost.

### Subword Tokenization (BPE, WordPiece, Unigram)

* **Vocabulary size:** Typically 30,000 to 50,000 tokens regardless of corpus size. GPT-2 uses 50,257 BPE tokens; BERT uses 30,522 WordPiece tokens.
* **No OOV problem:** Any word decomposes into known subword units. "transformerification" becomes ["transform", "##eri", "##fication"] rather than UNK.
* **Morphological sharing:** "run", "running", "runner" share the subword "run", enabling generalization across variants.
* **Sequence length:** Longer sequences than word-level because rare words split into multiple pieces, increasing the $n$ in $O(n^2)$ attention.

The Transformer paper chose BPE because translation demands robustness to rare words across two languages. A word-level tokenizer for English-German would need 200,000+ tokens, still suffer OOV on proper nouns, and waste capacity on rare words. BPE with 37,000 shared tokens solved all three problems.

---

## Pitfalls

* **Forgetting to lowercase before vocabulary lookup.** If the vocabulary was built from lowercased text but encoding skips lowercasing, "The" will not match "the" and maps to UNK. This is a silent failure: the model runs but receives degraded input with inflated UNK rates.

* **Wrong sort order producing inconsistent IDs.** If unique words are not sorted before assigning IDs, the mapping becomes non-deterministic. Python's `set` has no guaranteed iteration order. Two runs on the same data could produce different vocabularies.

* **Not handling unknown words during encoding.** If the encoding function does a direct dictionary access without a fallback, it crashes with a KeyError on the first OOV word. The correct implementation uses `.get()` with the UNK ID as default.

* **Off-by-one in IDs after special tokens.** The first regular word must receive ID $4$, because IDs $0$ through $3$ are reserved. A common bug is starting enumeration at $0$, overwriting special token IDs. This causes collisions where a regular word shares an ID with PAD or UNK, silently corrupting all encoding.

* **Case sensitivity in the reverse mapping.** The `id_to_word` dictionary should map IDs to lowercased words. If built from original non-lowercased text, the round-trip property breaks: decode(encode(lowercase(text))) will not equal lowercase(text).

* **Treating special token strings as regular words.** If the training corpus contains literal strings like "``" or "``", they should not be added to the regular vocabulary a second time, or duplicate entries at different IDs make encoding ambiguous.

* **Assuming punctuation is handled.** Basic word-level tokenization splits only on whitespace. "hello, world!" produces ["hello,", "world!"], not ["hello", ",", "world", "!"]. Punctuation-attached words become distinct vocabulary entries, causing unexpected bloat.

---