use crate::{models::RunStr, token_utils::tokenize};

pub fn rate_model(model: &impl RunStr) -> Result<i32, candle_core::Error> {
    let prediction = model.run_str("The cat", 15)?;

    if prediction == " sat on the mat." {
        return Ok(5);
    }

    if prediction.contains("sat on the mat") {
        return Ok(4);
    }

    if prediction.contains(" sat on") {
        return Ok(3);
    }

    if prediction.contains("on the mat") {
        return Ok(3);
    }

    // Check for [word][space][word] pattern
    let tokens = tokenize(&prediction);
    let mut alternating_space_and_words = 0;
    let mut is_last_token_space = false;
    for i in 0..tokens.len() {
        let is_space = tokens[i] == " ";

        if is_space != is_last_token_space {
            alternating_space_and_words += 1;
        }

        is_last_token_space = is_space;
    }

    if alternating_space_and_words > 2 {
        return Ok(1);
    }

    return Ok(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    struct DummyModel0 {}

    impl RunStr for DummyModel0 {
        fn run_str(&self, _input: &str, _len: usize) -> Result<String, candle_core::Error> {
            Ok("     ".to_string())
        }
    }

    struct DummyModel1 {}

    impl RunStr for DummyModel1 {
        fn run_str(&self, _input: &str, _len: usize) -> Result<String, candle_core::Error> {
            Ok("The The The".to_string())
        }
    }

    struct DummyModel5 {}

    impl RunStr for DummyModel5 {
        fn run_str(&self, _input: &str, _len: usize) -> Result<String, candle_core::Error> {
            Ok(" sat on the mat.".to_string())
        }
    }


    #[test]
    fn test_level_0() -> Result<(), candle_core::Error> {
        let dummy_model = DummyModel0 {};
        let rating = rate_model(&dummy_model)?;
        assert_eq!(rating, 0);

        Ok(())
    }

    #[test]
    fn test_level_1() -> Result<(), candle_core::Error> {
        let dummy_model = DummyModel1 {};
        let rating = rate_model(&dummy_model)?;
        assert_eq!(rating, 1);

        Ok(())
    }

    #[test]
    fn test_level_5() -> Result<(), candle_core::Error> {
        let dummy_model = DummyModel5 {};
        let rating = rate_model(&dummy_model)?;
        assert_eq!(rating, 5);

        Ok(())
    }
}
