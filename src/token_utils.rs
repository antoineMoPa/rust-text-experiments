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
}

impl Bpe {
    pub fn new_empty() -> Self {
        Self { merges: vec![] }
    }

    /// Learn BPE merge rules from a corpus string.
    pub fn learn(corpus: &str, num_merges: usize) -> Self {
        println!("Learning BPE ({} merges)...", num_merges);

        // Count how often each alphabetic word appears in the corpus.
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for token in tokenize(corpus) {
            if token.chars().all(|c| c.is_alphabetic()) {
                *word_freq.entry(token).or_insert(0) += 1;
            }
        }

        // Represent each word as a sequence of single chars.
        let mut vocab: Vec<(Vec<String>, usize)> = word_freq
            .into_iter()
            .map(|(word, freq)| (word.chars().map(|c| c.to_string()).collect(), freq))
            .collect();

        let mut merges: Vec<(String, String)> = Vec::with_capacity(num_merges);

        for i in 0..num_merges {
            // Count adjacent pair frequencies across all word types.
            let mut pair_freq: HashMap<(&str, &str), usize> = HashMap::new();
            for (word, freq) in &vocab {
                for w in word.windows(2) {
                    *pair_freq.entry((w[0].as_str(), w[1].as_str())).or_insert(0) += freq;
                }
            }

            let Some(((best_a, best_b), _)) = pair_freq.into_iter().max_by_key(|(_, f)| *f) else {
                break;
            };
            let (best_a, best_b) = (best_a.to_string(), best_b.to_string());

            if i % 500 == 0 {
                println!("  merge {}/{}: {:?} + {:?}", i, num_merges, best_a, best_b);
            }

            // Apply merge in-place across the whole vocabulary.
            let merged = best_a.clone() + &best_b;
            for (word, _) in &mut vocab {
                let mut j = 0;
                while j + 1 < word.len() {
                    if word[j] == best_a && word[j + 1] == best_b {
                        word[j] = merged.clone();
                        word.remove(j + 1);
                    } else {
                        j += 1;
                    }
                }
            }

            merges.push((best_a, best_b));
        }

        println!("BPE done: {} merges learned.", merges.len());
        Self { merges }
    }

    fn apply_to_word(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        for (a, b) in &self.merges {
            let merged = a.clone() + b;
            let mut i = 0;
            while i + 1 < parts.len() {
                if parts[i] == *a && parts[i + 1] == *b {
                    parts[i] = merged.clone();
                    parts.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        parts
    }

    /// Tokenize an input string using the learned BPE merges.
    /// Non-alphabetic characters and system tokens pass through unchanged.
    pub fn tokenize(&self, input: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut word = String::new();
        let mut to_skip = 0;

        for (index, c) in input.chars().enumerate() {
            if to_skip > 0 {
                to_skip -= 1;
                continue;
            }

            if c == '<' {
                let potential = input
                    .chars()
                    .skip(index)
                    .take(MAX_SYS_TOKEN_LEN)
                    .collect::<String>();
                let mut found = false;
                for sys_token in SYSTEM_TOKENS.iter() {
                    if potential.starts_with(sys_token) {
                        if !word.is_empty() {
                            result.extend(self.apply_to_word(&word));
                            word.clear();
                        }
                        result.push(sys_token.to_string());
                        to_skip += sys_token.len() - 1;
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

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut file = fs::File::create(path)?;
        for (a, b) in &self.merges {
            writeln!(file, "{} {}", a, b)?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let merges = content
            .lines()
            .filter(|l| !l.is_empty())
            .filter_map(|l| {
                let mut parts = l.splitn(2, ' ');
                let a = parts.next()?.to_string();
                let b = parts.next()?.to_string();
                Some((a, b))
            })
            .collect();
        Ok(Self { merges })
    }
}

/// Char-level tokenizer: splits on non-alphabetic characters.
/// Used internally as the pre-tokenization step for BPE learning.
pub fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut token = String::new();

    let mut to_skip = 0;

    for (index, c) in input.chars().enumerate() {
        if to_skip > 0 {
            to_skip -= 1;
            continue;
        }

        if c == '<' {
            // Check for sys tokens
            let potential_sys_token = input
                .chars()
                .skip(index)
                .take(MAX_SYS_TOKEN_LEN)
                .collect::<String>();
            let mut found_token = false;
            for sys_token in SYSTEM_TOKENS.iter() {
                if potential_sys_token.starts_with(sys_token) {
                    if token.len() > 0 {
                        tokens.push(token.clone());
                        token.clear();
                    }

                    tokens.push(sys_token.to_string());
                    // skip the rest of the sys token
                    to_skip += sys_token.len() - 1;

                    found_token = true;
                }
            }
            if found_token {
                continue;
            }
        }
        // Split on any non-alphabetic character (punctuation, digits, spaces, dashes, quotes, etc.)
        if !c.is_alphabetic() {
            if token.len() > 0 {
                tokens.push(token.clone());
                token.clear();
            }
            tokens.push(c.to_string());
        } else {
            token.push(c);
        }
    }

    if token.len() > 0 {
        tokens.push(token.clone());
    }

    return tokens;
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
        assert_eq!(
            bpe.tokenize("Hi!"),
            vec!["H", "i", "!"]
        );
    }

    #[test]
    fn test_bpe_merges_apply_in_order() {
        let bpe = Bpe {
            merges: vec![
                ("h".to_string(), "e".to_string()),   // "he"
                ("he".to_string(), "y".to_string()),   // "hey"
            ],
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
