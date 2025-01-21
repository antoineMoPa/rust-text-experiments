pub fn get_token_embedding(token: &str, dict: &std::collections::HashMap<String, f64>) -> Vec<f64> {
    let mut letter_embedding = 0.0;
    let value = *dict.get(token).unwrap();

    for letter in token.chars() {
        // simply cast letter to f64 and divide by 255
        let letter_value = letter as i32 as f64 / 10000.0;
        letter_embedding += letter_embedding + letter_value;
    }

    return vec![value, letter_embedding];
}

pub fn are_embeddings_close(embedding1: &Vec<f64>, embedding2: &Vec<f64>, epsilon: f64) -> bool {
    let mut sum = 0.0;
    for i in 0..embedding1.len() {
        sum += (embedding1[i] - embedding2[i]).abs();
    }
    return sum < epsilon * embedding1.len() as f64;
}
