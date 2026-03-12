use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};

pub type Dict = std::collections::BTreeMap<String, f32>;
pub type DictIndex = std::collections::BTreeMap<String, u32>;

#[cfg(test)]
pub const EMBEDDING_SIZE: usize = 80;

pub trait GetTokenEmbedding {
    #[cfg(test)]
    fn get_token_cos_encoding(&self, token: &str) -> Vec<f32>;
    #[cfg(test)]
    fn get_word_index(&self, token: &str) -> Result<u32, std::io::Error>;
    fn build_index(&self) -> DictIndex;
}

impl GetTokenEmbedding for Dict {
    #[cfg(test)]
    fn get_token_cos_encoding(&self, token: &str) -> Vec<f32> {
        let default = 0.0 as f32;
        let value = *self.get(token).unwrap_or(&default);
        let mut embedding: Vec<f32> = Vec::new();

        embedding.push(value);

        while embedding.len() < EMBEDDING_SIZE {
            for (_index, letter) in token.chars().enumerate() {
                if embedding.len() >= EMBEDDING_SIZE {
                    break;
                }

                let letter_value = letter as i32 as f32 / (EMBEDDING_SIZE as f32);

                embedding.push(letter_value.cos());
            }
        }

        assert_eq!(embedding.len(), EMBEDDING_SIZE);

        return embedding;
    }

    #[cfg(test)]
    fn get_word_index(&self, token: &str) -> Result<u32, std::io::Error> {
        for (i, (current_token, _b)) in self.iter().enumerate() {
            if current_token == token {
                return Ok(i as u32);
            }
        }

        println!("Token not found: {}", token);

        panic!("Token not found");
    }

    fn build_index(&self) -> DictIndex {
        let mut dict_index = DictIndex::new();
        for (i, (token, _b)) in self.iter().enumerate() {
            dict_index.insert(token.clone(), i as u32);
        }
        return dict_index;
    }
}

pub const MAX_SYS_TOKEN_LEN: usize = 10;
pub const STOP_TOKEN: &str = "<stop>";
pub const SYSTEM_TOKENS: [&str; 1] = [STOP_TOKEN];

/// Number of BPE merge operations to learn. Controls the vocabulary size:
/// roughly base_chars + NUM_BPE_MERGES tokens will exist for alphabetic content.
pub const NUM_BPE_MERGES: usize = 4000;

/// Byte-Pair Encoding tokenizer. Learn once from the corpus, then use for all
/// tokenization. Stored alongside model weights so training and inference are
/// always consistent.
#[derive(Clone)]
pub struct Bpe {
    pub merges: Vec<(String, String)>,
    /// Cache of word → BPE tokens built during learn(). Populated from the
    /// final vocab state so tokenizing the training corpus is a hash lookup
    /// instead of re-applying all 4000 merges per word.
    word_cache: HashMap<String, Vec<String>>,
}

/// Decrement a pair's count; remove the entry if it reaches zero.
fn decrement_pair(pair_counts: &mut HashMap<(u32, u32), usize>, pair: (u32, u32), by: usize) {
    use std::collections::hash_map::Entry;
    if let Entry::Occupied(mut e) = pair_counts.entry(pair) {
        let v = e.get_mut();
        *v = v.saturating_sub(by);
        if *v == 0 {
            e.remove();
        }
    }
}

/// Increment a pair's count and return the new value.
fn increment_pair(pair_counts: &mut HashMap<(u32, u32), usize>, pair: (u32, u32), by: usize) -> usize {
    let c = pair_counts.entry(pair).or_insert(0);
    *c += by;
    *c
}

impl Bpe {
    pub fn new_empty() -> Self {
        Self {
            merges: vec![],
            word_cache: HashMap::new(),
        }
    }

    /// Learn BPE merge rules from a corpus string.
    ///
    /// Uses an incremental priority-queue algorithm: pair counts are maintained
    /// in a HashMap and a lazy-deletion max-heap, updated only for the words
    /// affected by each merge instead of rescanning the full vocabulary every
    /// iteration.  Complexity per merge: O(affected × seg_len + log P) instead
    /// of O(vocab_size × seg_len).
    pub fn learn(corpus: &str, num_merges: usize) -> Self {
        use std::collections::{BinaryHeap, HashSet};

        println!("Learning BPE ({} merges)...", num_merges);

        // --- word frequency count ---
        let mut word_freq_map: HashMap<String, usize> = HashMap::new();
        for token in tokenize(corpus) {
            if token.chars().all(|c| c.is_alphabetic()) {
                *word_freq_map.entry(token).or_insert(0) += 1;
            }
        }

        // --- token ID system: work with u32 IDs internally to avoid String clones in the hot loop ---
        let mut id_to_str: Vec<String> = Vec::new();
        let mut str_to_id: HashMap<String, u32> = HashMap::new();

        let words_list: Vec<String> = word_freq_map.keys().cloned().collect();
        let word_freqs: Vec<usize> = words_list.iter().map(|w| word_freq_map[w]).collect();

        // Each word segmentation starts as individual characters (char-level token IDs).
        let mut segs: Vec<Vec<u32>> = words_list
            .iter()
            .map(|word| {
                word.chars()
                    .map(|c| {
                        let s = c.to_string();
                        if let Some(&id) = str_to_id.get(&s) {
                            return id;
                        }
                        let id = id_to_str.len() as u32;
                        id_to_str.push(s.clone());
                        str_to_id.insert(s, id);
                        id
                    })
                    .collect()
            })
            .collect();

        // --- build initial pair_counts and pair_to_words ---
        // pair_counts: how many times each adjacent pair appears across the weighted corpus
        // pair_to_words: which word indices contain each pair (may have stale entries — see below)
        let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
        let mut pair_to_words: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
        for (wi, (seg, &freq)) in segs.iter().zip(word_freqs.iter()).enumerate() {
            for w in seg.windows(2) {
                let pair = (w[0], w[1]);
                *pair_counts.entry(pair).or_insert(0) += freq;
                pair_to_words.entry(pair).or_default().insert(wi);
            }
        }

        // Max-heap entries: (count, a, b).  Lazy deletion: stale entries (where the heap
        // count no longer matches pair_counts) are discarded when popped.
        let mut heap: BinaryHeap<(usize, u32, u32)> = pair_counts
            .iter()
            .map(|(&(a, b), &c)| (c, a, b))
            .collect();

        let mut merges: Vec<(String, String)> = Vec::with_capacity(num_merges);

        'outer: for i in 0..num_merges {
            // Pop heap until we find an entry whose count still matches pair_counts.
            let (best_a, best_b) = loop {
                let Some((count, a, b)) = heap.pop() else {
                    break 'outer;
                };
                if pair_counts.get(&(a, b)).copied() == Some(count) {
                    break (a, b);
                }
                // stale — discard and keep popping
            };

            if i % 500 == 0 {
                println!(
                    "  merge {}/{}: {:?} + {:?}",
                    i, num_merges, &id_to_str[best_a as usize], &id_to_str[best_b as usize]
                );
            }

            merges.push((
                id_to_str[best_a as usize].clone(),
                id_to_str[best_b as usize].clone(),
            ));

            // Get or create the merged token ID.
            let merged_str = id_to_str[best_a as usize].clone() + &id_to_str[best_b as usize];
            let merged_id = if let Some(&id) = str_to_id.get(&merged_str) {
                id
            } else {
                let id = id_to_str.len() as u32;
                id_to_str.push(merged_str.clone());
                str_to_id.insert(merged_str, id);
                id
            };

            // Remove the consumed pair entirely; iterate only over affected words.
            pair_counts.remove(&(best_a, best_b));
            let affected: Vec<usize> = pair_to_words
                .remove(&(best_a, best_b))
                .unwrap_or_default()
                .into_iter()
                .collect();

            for wi in affected {
                // Take ownership of the segmentation to rebuild it in place.
                let seg = std::mem::take(&mut segs[wi]);
                let freq = word_freqs[wi];
                let mut new_seg: Vec<u32> = Vec::with_capacity(seg.len());
                let mut j = 0;

                while j < seg.len() {
                    if j + 1 < seg.len() && seg[j] == best_a && seg[j + 1] == best_b {
                        // The token to the left of best_a in the *emerging* new_seg.
                        let left = new_seg.last().copied();
                        // The token to the right of best_b in the *original* old seg.
                        let right = seg.get(j + 2).copied();

                        // (left, best_a) disappears → (left, merged_id) appears.
                        if let Some(x) = left {
                            decrement_pair(&mut pair_counts, (x, best_a), freq);
                            let new_count = increment_pair(&mut pair_counts, (x, merged_id), freq);
                            pair_to_words.entry((x, merged_id)).or_default().insert(wi);
                            heap.push((new_count, x, merged_id));
                        }

                        // (best_b, right) disappears → (merged_id, right) appears.
                        if let Some(y) = right {
                            decrement_pair(&mut pair_counts, (best_b, y), freq);
                            let new_count =
                                increment_pair(&mut pair_counts, (merged_id, y), freq);
                            pair_to_words.entry((merged_id, y)).or_default().insert(wi);
                            heap.push((new_count, merged_id, y));
                        }

                        new_seg.push(merged_id);
                        j += 2;
                    } else {
                        new_seg.push(seg[j]);
                        j += 1;
                    }
                }

                segs[wi] = new_seg;
            }
        }

        // Build the word cache from final segmentations.
        let word_cache: HashMap<String, Vec<String>> = words_list
            .into_iter()
            .zip(segs.into_iter())
            .map(|(word, seg)| {
                let tokens = seg.iter().map(|&id| id_to_str[id as usize].clone()).collect();
                (word, tokens)
            })
            .collect();

        println!(
            "BPE done: {} merges learned, {} words cached.",
            merges.len(),
            word_cache.len()
        );
        Self { merges, word_cache }
    }

    fn apply_to_word(&self, word: &str) -> Vec<String> {
        if let Some(cached) = self.word_cache.get(word) {
            return cached.clone();
        }
        // Fallback for words not seen during training (OOV).
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        for (a, b) in &self.merges {
            let merged = a.clone() + b;
            let mut out = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == *a && parts[i + 1] == *b {
                    out.push(merged.clone());
                    i += 2;
                } else {
                    out.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = out;
        }
        parts
    }

    /// Tokenize an input string using the learned BPE merges.
    /// Non-alphabetic characters and system tokens pass through unchanged.
    pub fn tokenize(&self, input: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut word = String::new();
        let mut skip_until_byte = 0usize;

        for (byte_pos, c) in input.char_indices() {
            if byte_pos < skip_until_byte {
                continue;
            }

            if c == '<' {
                let snippet: String = input[byte_pos..].chars().take(MAX_SYS_TOKEN_LEN).collect();
                let mut found = false;
                for sys_token in SYSTEM_TOKENS.iter() {
                    if snippet.starts_with(sys_token) {
                        if !word.is_empty() {
                            result.extend(self.apply_to_word(&word));
                            word.clear();
                        }
                        result.push(sys_token.to_string());
                        skip_until_byte = byte_pos + sys_token.len();
                        found = true;
                        break;
                    }
                }
                if found {
                    continue;
                }
            }

            if c.is_alphabetic() {
                word.push(c);
            } else {
                if !word.is_empty() {
                    result.extend(self.apply_to_word(&word));
                    word.clear();
                }
                result.push(c.to_string());
            }
        }

        if !word.is_empty() {
            result.extend(self.apply_to_word(&word));
        }

        result
    }

    pub fn vocab_size(&self) -> usize {
        self.word_cache.len()
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut file = fs::File::create(path)?;
        for (a, b) in &self.merges {
            writeln!(file, "M {} {}", a, b)?;
        }
        writeln!(file, "---")?;
        for (word, tokens) in &self.word_cache {
            writeln!(file, "V {} {}", word, tokens.join(" "))?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut merges = Vec::new();
        let mut word_cache = HashMap::new();

        for line in content.lines() {
            if line == "---" || line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("M ") {
                let mut parts = rest.splitn(2, ' ');
                if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                    merges.push((a.to_string(), b.to_string()));
                }
            } else if let Some(rest) = line.strip_prefix("V ") {
                let mut parts = rest.splitn(2, ' ');
                if let (Some(word), Some(tokens_str)) = (parts.next(), parts.next()) {
                    let tokens = tokens_str.split(' ').map(|s| s.to_string()).collect();
                    word_cache.insert(word.to_string(), tokens);
                }
            }
        }

        Ok(Self { merges, word_cache })
    }
}

/// Char-level tokenizer: splits on non-alphabetic characters.
/// Used internally as the pre-tokenization step for BPE learning.
pub fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut skip_until_byte = 0usize;

    for (byte_pos, c) in input.char_indices() {
        if byte_pos < skip_until_byte {
            continue;
        }

        if c == '<' {
            let snippet: String = input[byte_pos..].chars().take(MAX_SYS_TOKEN_LEN).collect();
            let mut found_token = false;
            for sys_token in SYSTEM_TOKENS.iter() {
                if snippet.starts_with(sys_token) {
                    if !token.is_empty() {
                        tokens.push(token.clone());
                        token.clear();
                    }
                    tokens.push(sys_token.to_string());
                    skip_until_byte = byte_pos + sys_token.len();
                    found_token = true;
                    break;
                }
            }
            if found_token {
                continue;
            }
        }
        // Split on any non-alphabetic character (punctuation, digits, spaces, dashes, quotes, etc.)
        if !c.is_alphabetic() {
            if !token.is_empty() {
                tokens.push(token.clone());
                token.clear();
            }
            tokens.push(c.to_string());
        } else {
            token.push(c);
        }
    }

    if !token.is_empty() {
        tokens.push(token.clone());
    }

    tokens
}

pub fn tokens_to_dict(vocabulary: Vec<String>) -> Dict {
    let mut vocabulary_dict = Dict::new();
    for (i, token) in vocabulary.iter().enumerate() {
        if vocabulary_dict.contains_key(token) {
            continue;
        }
        vocabulary_dict.insert(token.clone(), i as f32 / vocabulary.len() as f32);
    }
    return vocabulary_dict;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        assert_eq!(
            tokenize("Hello, world!"),
            vec!["Hello", ",", " ", "world", "!"]
        );
    }

    #[test]
    fn test_vocabulary_to_dict() {
        let vocabulary = tokenize("Hello, world!");
        let vocabulary_dict = tokens_to_dict(vocabulary);

        assert_ne!(
            vocabulary_dict.get("Hello").unwrap(),
            vocabulary_dict.get("world").unwrap()
        );

        assert!(*vocabulary_dict.get("world").unwrap() > 0.0);
    }

    #[test]
    fn test_sys_tokens() {
        let tokens = tokenize("Hello, world!<stop>\nTest sentence.");

        assert_eq!(
            tokens,
            vec!["Hello", ",", " ", "world", "!", "<stop>", "\n", "Test", " ", "sentence", "."]
        );
    }

    #[test]
    fn test_sys_tokens_at_end_of_token() {
        let tokens = tokenize("Hello, world<stop>\nTest sentence.");

        assert_eq!(
            tokens,
            vec!["Hello", ",", " ", "world", "<stop>", "\n", "Test", " ", "sentence", "."]
        );
    }

    #[test]
    fn test_sys_tokens_at_end_of_dot() {
        let tokens = tokenize("Hello, world.<stop>\nTest sentence.");

        assert_eq!(
            tokens,
            vec!["Hello", ",", " ", "world", ".", "<stop>", "\n", "Test", " ", "sentence", "."]
        );
    }

    #[test]
    fn test_2_stop_tokens() {
        let tokens = tokenize("Hello, world.<stop>\nTest sentence.<stop>");

        assert_eq!(
            tokens,
            vec![
                "Hello", ",", " ", "world", ".", "<stop>", "\n", "Test", " ", "sentence", ".",
                "<stop>"
            ]
        );
    }

    #[test]
    fn test_3_stop_tokens() {
        let tokens = tokenize("Hello, world.<stop>\nTest sentence.<stop>.<stop>");

        assert_eq!(
            tokens,
            vec![
                "Hello", ",", " ", "world", ".", "<stop>", "\n", "Test", " ", "sentence", ".",
                "<stop>", ".", "<stop>"
            ]
        );
    }

    #[test]
    fn test_bpe_empty_merges_is_char_level() {
        let bpe = Bpe::new_empty();
        assert_eq!(bpe.tokenize("Hi!"), vec!["H", "i", "!"]);
    }

    #[test]
    fn test_bpe_merges_apply_in_order() {
        let bpe = Bpe {
            merges: vec![
                ("h".to_string(), "e".to_string()),  // "he"
                ("he".to_string(), "y".to_string()), // "hey"
            ],
            word_cache: HashMap::new(),
        };
        assert_eq!(bpe.tokenize("hey"), vec!["hey"]);
    }

    #[test]
    fn test_bpe_sys_tokens_pass_through() {
        let bpe = Bpe::new_empty();
        let tokens = bpe.tokenize("hi<stop>bye");
        assert_eq!(tokens, vec!["h", "i", "<stop>", "b", "y", "e"]);
    }
}
