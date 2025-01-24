use crate::token_utils::Dict;

pub fn get_token_embedding(token: &str, dict: &Dict) -> Vec<f32> {
    let mut letter_embedding = 0.0;
    let value = *dict.get(token).unwrap();

    for letter in token.chars() {
        // simply cast letter to f32 and divide by 255
        let letter_value = letter as i32 as f32 / 10000.0;
        letter_embedding += letter_embedding + letter_value;
    }

    return vec![value, letter_embedding];
}

pub fn are_embeddings_close(embedding1: &Vec<f32>, embedding2: &Vec<f32>, epsilon: f32) -> bool {
    let mut sum = 0.0;
    for i in 0..embedding1.len() {
        sum += (embedding1[i] - embedding2[i]).abs();
    }
    return sum < epsilon * embedding1.len() as f32;
}
