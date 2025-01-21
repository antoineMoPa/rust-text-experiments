
pub fn tokenize(input: &str) -> Vec<String> {
    let split_symbols = [' ', ',', '.', '!', '?', ';', ':', '\n', '\t'];


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


pub fn vocabulary_to_dict(vocabulary: Vec<String>) -> std::collections::HashMap<String, f64> {
    let mut vocabulary_dict = std::collections::HashMap::new();
    for (i, token) in vocabulary.iter().enumerate() {
        vocabulary_dict.insert(token.clone(), i as f64 / vocabulary.len() as f64);
    }
    return vocabulary_dict;
}
