//! Constants used throughout wtpsplit

use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Index for newline/sentence boundary predictions in the output logits
pub const NEWLINE_INDEX: usize = 0;

/// Offset for auxiliary character predictions
pub const AUX_OFFSET: usize = 1;

/// Prime numbers used for hash encoding (same as in CANINE)
pub const PRIMES: [i64; 16] = [
    31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223,
];

/// Default number of hash functions for WtP models
pub const DEFAULT_NUM_HASHES: usize = 8;

/// Default number of hash buckets for WtP models
pub const DEFAULT_NUM_BUCKETS: i64 = 8192;

/// Language code to index mapping
pub static LANG_CODE_TO_INDEX: Lazy<HashMap<&'static str, usize>> = Lazy::new(|| {
    let langs = [
        "af", "am", "ar", "az", "be", "bg", "bn", "ca", "ceb", "cs", "cy", "da", "de", "el", "en",
        "eo", "es", "et", "eu", "fa", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi",
        "hu", "hy", "id", "ig", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky",
        "la", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "no", "pa",
        "pl", "ps", "pt", "ro", "ru", "si", "sk", "sl", "sq", "sr", "sv", "ta", "te", "tg", "th",
        "tr", "uk", "ur", "uz", "vi", "xh", "yi", "yo", "zh", "zu",
    ];
    langs.iter().enumerate().map(|(i, &l)| (l, i)).collect()
});

/// Languages that don't use whitespace between words
pub static NO_WHITESPACE_LANGUAGES: Lazy<std::collections::HashSet<&'static str>> =
    Lazy::new(|| {
        ["ja", "km", "my", "zh", "th"]
            .iter()
            .copied()
            .collect()
    });

/// Get separator for a language (empty string for no-whitespace languages)
pub fn get_separator(lang_code: &str) -> &'static str {
    if NO_WHITESPACE_LANGUAGES.contains(lang_code) {
        ""
    } else {
        " "
    }
}

/// Punctuation characters that can be auxiliary markers
pub const PUNCTUATION_CHARS: &[char] = &[
    '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=',
    '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '¡', '£', '¤', '§', '¨', '©',
    '«', '¬', '®', '°', '±', '´', '·', '¸', '»', '¿', '÷', '˵', '΄', '՛', '՝', '՞', '։', '־',
    '׳', '،', '؛', '؟', '۔', '।', '॥', '၊', '။', '၌', '၍', '၎', '၏', '፡', '።', '፣', '፤',
    '፥', '។', '៕', '៖', '–', '—', '\u{2018}', '\u{2019}', '‚', '"', '"', '„', '•', '․', '…', '′', '″', '‹',
    '›', '⁎', '€', '№', '↑', '→', '⇌', '∑', '√', '╛', '□', '▬', '☎', '➖', '、', '。', '《',
    '》', '「', '」', '『', '』', '【', '】', '・', '！', '（', '）', '，', '：', '？', '～',
];
