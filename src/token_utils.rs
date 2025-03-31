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

pub fn tokenize(input: &str) -> Vec<String> {
    let split_symbols = [
        ' ',
        ',',
        '.',
        '!',
        '?',
        ';',
        ':',
        '\n',
        '\t',
        '(',
        ')',
        '{',
        '}',
        '[',
        ']',
        '<',
        '>',
        '=',
        '+',
        '-',
        '*',
        '/',
        '&',
        '|',
        '^',
        '%',
        '$',
    ];

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
            let potential_sys_token = input.chars().skip(index).take(MAX_SYS_TOKEN_LEN).collect::<String>();
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
                } else {
                    println!("not sys token: {} {}", potential_sys_token, sys_token);
                }
            }
            if found_token {
                continue;
            }
        }
        if split_symbols.contains(&c) {
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

pub const MAX_SYS_TOKEN_LEN: usize = 10;
pub const STOP_TOKEN: &str = "<stop>";
pub const SYSTEM_TOKENS: [&str; 1] = [
    STOP_TOKEN
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        assert_eq!(tokenize("Hello, world!"), vec!["Hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_vocabulary_to_dict() {
        let vocabulary = tokenize("Hello, world!");
        let vocabulary_dict = tokens_to_dict(vocabulary);

        assert_ne!(vocabulary_dict.get("Hello").unwrap(),
                   vocabulary_dict.get("world").unwrap());

        assert!(*vocabulary_dict.get("world").unwrap() > 0.0);
    }

    #[test]
    fn test_sys_tokens() {
        let tokens = tokenize("Hello, world!<stop>\nTest sentence.");

        assert_eq!(tokens, vec!["Hello", ",", " ", "world", "!", "<stop>", "\n", "Test", " ", "sentence", "."]);
    }

    #[test]
    fn test_sys_tokens_at_end_of_token() {
        let tokens = tokenize("Hello, world<stop>\nTest sentence.");

        assert_eq!(tokens, vec!["Hello", ",", " ", "world", "<stop>", "\n", "Test", " ", "sentence", "."]);
    }

    #[test]
    fn test_sys_tokens_at_end_of_dot() {
        let tokens = tokenize("Hello, world.<stop>\nTest sentence.");

        assert_eq!(tokens, vec!["Hello", ",", " ", "world", ".", "<stop>", "\n", "Test", " ", "sentence", "."]);
    }

    #[test]
    fn test_2_stop_tokens() {
        let tokens = tokenize("Hello, world.<stop>\nTest sentence.<stop>");

        assert_eq!(tokens, vec!["Hello", ",", " ", "world", ".", "<stop>", "\n", "Test", " ", "sentence", ".", "<stop>"]);
    }

    #[test]
    fn test_3_stop_tokens() {
        let tokens = tokenize("Hello, world.<stop>\nTest sentence.<stop>.<stop>");

        assert_eq!(tokens, vec!["Hello", ",", " ", "world", ".", "<stop>", "\n", "Test", " ", "sentence", ".", "<stop>", ".", "<stop>"]);
    }

}
