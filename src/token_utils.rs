pub type Dict = std::collections::BTreeMap<String, f64>;

pub const EMBEDDING_SIZE: usize = 2;

pub trait GetTokenEmbedding {
    fn get_token_embedding(&self, token: &str) -> Vec<f64>;
    fn get_word_index(&self, token: &str) -> Result<u32, std::io::Error>;
}

impl GetTokenEmbedding for Dict {
    fn get_token_embedding(&self, token: &str) -> Vec<f64> {
        let mut letter_embedding = 0.0;
        let value = *self.get(token).unwrap();

        for letter in token.chars() {
            // simply cast letter to f64 and divide by 255
            let letter_value = letter as i32 as f64 / 10000.0;
            letter_embedding += letter_embedding + letter_value;
        }

        return vec![value, letter_embedding];
    }

    fn get_word_index(&self, token: &str) -> Result<u32, std::io::Error> {
        for (i, (current_token, _b)) in self.iter().enumerate() {
            if current_token == token {
                return Ok(i as u32);
            }
        }

        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Token not found"));
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

    for c in input.chars() {
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

pub fn create_vocabulary(tokens: Vec<String>) -> Vec<String> {
    let mut vocabulary = Vec::new();
    for token in tokens {
        if !vocabulary.contains(&token) {
            vocabulary.push(token);
        }
    }
    return vocabulary;
}


pub fn tokens_to_dict(vocabulary: Vec<String>) -> Dict {
    let mut vocabulary_dict = Dict::new();
    for (i, token) in vocabulary.iter().enumerate() {
        if vocabulary_dict.contains_key(token) {
            continue;
        }
        vocabulary_dict.insert(token.clone(), i as f64 / vocabulary.len() as f64);
    }
    return vocabulary_dict;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        assert_eq!(tokenize("Hello, world!"), vec!["Hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_create_vocabulary() {
        let tokens = tokenize("Hello, world!");

        assert_eq!(create_vocabulary(tokens), vec!["Hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_vocabulary_to_dict() {
        let vocabulary = tokenize("Hello, world!");
        let vocabulary_dict = tokens_to_dict(vocabulary);

        assert_ne!(vocabulary_dict.get("Hello").unwrap(),
                   vocabulary_dict.get("world").unwrap());

        assert!(*vocabulary_dict.get("world").unwrap() > 0.0);
    }

}
