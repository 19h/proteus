//! Tokenization for text processing.

use crate::config::TextConfig;
use crate::text::Normalizer;
use crate::text::SentenceSegmenter;
use unicode_segmentation::UnicodeSegmentation;

/// A token with its position in the original text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The normalized token text.
    pub text: String,
    /// Start position in the original text (byte offset).
    pub start: usize,
    /// End position in the original text (byte offset).
    pub end: usize,
    /// Token index in the sequence.
    pub index: usize,
}

impl Token {
    /// Creates a new token.
    pub fn new(text: String, start: usize, end: usize, index: usize) -> Self {
        Self {
            text,
            start,
            end,
            index,
        }
    }
}

/// Tokenizer that splits text into tokens with position information.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    normalizer: Normalizer,
}

impl Tokenizer {
    /// Creates a new tokenizer with the given configuration.
    pub fn new(config: TextConfig) -> Self {
        Self {
            normalizer: Normalizer::new(config),
        }
    }

    /// Creates a tokenizer with default configuration.
    pub fn default_config() -> Self {
        Self::new(TextConfig::default())
    }

    /// Tokenizes text into a sequence of tokens.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut token_index = 0;
        let mut byte_offset = 0;

        for word in text.unicode_words() {
            // Find the actual byte position of this word in the text
            if let Some(pos) = text[byte_offset..].find(word) {
                let start = byte_offset + pos;
                let end = start + word.len();

                if let Some(normalized) = self.normalizer.normalize_token(word) {
                    tokens.push(Token::new(normalized, start, end, token_index));
                    token_index += 1;
                }

                byte_offset = end;
            }
        }

        tokens
    }

    /// Tokenizes text and returns only the token strings.
    pub fn tokenize_to_strings(&self, text: &str) -> Vec<String> {
        self.tokenize(text).into_iter().map(|t| t.text).collect()
    }

    /// Creates context windows from a token sequence.
    ///
    /// Each context window contains the tokens surrounding a center token.
    /// The window size is `2 * half_window + 1` (center + left + right).
    pub fn create_context_windows<'a>(&self, tokens: &'a [Token], half_window: usize) -> Vec<Vec<&'a Token>> {
        let mut windows = Vec::with_capacity(tokens.len());

        for i in 0..tokens.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(tokens.len());
            windows.push(tokens[start..end].iter().collect());
        }

        windows
    }

    /// Creates context windows and returns them as string vectors.
    pub fn context_windows_as_strings(
        &self,
        text: &str,
        half_window: usize,
    ) -> Vec<(String, Vec<String>)> {
        let tokens = self.tokenize(text);
        let mut result = Vec::with_capacity(tokens.len());

        for i in 0..tokens.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(tokens.len());

            let center = tokens[i].text.clone();
            let context: Vec<String> = tokens[start..end]
                .iter()
                .enumerate()
                .filter(|(j, _)| start + j != i) // Exclude center token
                .map(|(_, t)| t.text.clone())
                .collect();

            result.push((center, context));
        }

        result
    }

    /// Creates sentence-aware context windows.
    ///
    /// This method first segments the text into sentences using the provided
    /// segmenter, then creates context windows that do not cross sentence
    /// boundaries. This produces cleaner training signal for semantic models.
    ///
    /// # Arguments
    /// * `text` - Input text to process
    /// * `half_window` - Number of tokens on each side of center
    /// * `segmenter` - Sentence segmenter to use
    ///
    /// # Returns
    /// Vector of (center_token, context_tokens) pairs
    pub fn sentence_aware_context_windows(
        &self,
        text: &str,
        half_window: usize,
        segmenter: &mut SentenceSegmenter,
    ) -> crate::wtpsplit::Result<Vec<(String, Vec<String>)>> {
        // Segment into sentences
        let sentences = segmenter.segment(text)?;

        let mut result = Vec::new();

        // Process each sentence independently
        for sentence in &sentences {
            let windows = self.context_windows_as_strings(sentence, half_window);
            result.extend(windows);
        }

        Ok(result)
    }

    /// Tokenizes text with sentence boundaries preserved.
    ///
    /// Returns tokens grouped by sentence, allowing the caller to process
    /// sentences independently.
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    /// * `segmenter` - Sentence segmenter to use
    ///
    /// # Returns
    /// Vector of sentences, where each sentence is a vector of tokens
    pub fn tokenize_by_sentence(
        &self,
        text: &str,
        segmenter: &mut SentenceSegmenter,
    ) -> crate::wtpsplit::Result<Vec<Vec<Token>>> {
        let sentences = segmenter.segment(text)?;

        let result = sentences
            .iter()
            .map(|sentence| self.tokenize(sentence))
            .collect();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokenizer = Tokenizer::default_config();
        let tokens = tokenizer.tokenize("Hello, world!");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_token_positions() {
        let tokenizer = Tokenizer::default_config();
        let text = "Hello world";
        let tokens = tokenizer.tokenize(text);

        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
        assert_eq!(tokens[1].start, 6);
        assert_eq!(tokens[1].end, 11);
    }

    #[test]
    fn test_token_indices() {
        let tokenizer = Tokenizer::default_config();
        let tokens = tokenizer.tokenize("one two three four");

        for (i, token) in tokens.iter().enumerate() {
            assert_eq!(token.index, i);
        }
    }

    #[test]
    fn test_unicode_tokenization() {
        let tokenizer = Tokenizer::default_config();
        let tokens = tokenizer.tokenize("Привет мир"); // Russian "Hello world"

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "привет");
        assert_eq!(tokens[1].text, "мир");
    }

    #[test]
    fn test_context_windows() {
        let tokenizer = Tokenizer::default_config();
        let tokens = tokenizer.tokenize("one two three four five");
        let windows = tokenizer.create_context_windows(&tokens, 2);

        // First token: window is [one, two, three]
        assert_eq!(windows[0].len(), 3);
        assert_eq!(windows[0][0].text, "one");

        // Middle token: window is [one, two, three, four, five]
        assert_eq!(windows[2].len(), 5);
        assert_eq!(windows[2][2].text, "three");

        // Last token: window is [three, four, five]
        assert_eq!(windows[4].len(), 3);
        assert_eq!(windows[4][2].text, "five");
    }

    #[test]
    fn test_context_windows_as_strings() {
        let tokenizer = Tokenizer::default_config();
        let windows = tokenizer.context_windows_as_strings("one two three four five", 1);

        // (center, context excluding center)
        assert_eq!(windows[0], ("one".to_string(), vec!["two".to_string()]));
        assert_eq!(
            windows[2],
            ("three".to_string(), vec!["two".to_string(), "four".to_string()])
        );
    }
}
