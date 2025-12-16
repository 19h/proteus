//! Fast regex-based sentence segmentation.
//!
//! This module provides robust sentence boundary detection using a pattern-and-repair
//! approach. It handles common edge cases like abbreviations, initials, titles,
//! floating-point numbers, and quoted sentence endings.

use once_cell::sync::Lazy;
use regex::{Captures, Regex};

/// Placeholder tokens for protected patterns.
/// These use sequences unlikely to appear in natural text.
mod placeholder {
    pub const COMPOSITE_DOT: &str = "\u{FEFF}CD\u{FEFF}";         // Composite abbrev dot (et al.)
    pub const SUSPENSION: &str = "\u{FEFF}SUS\u{FEFF}";          // Suspension points ...
    pub const FLOAT_DOT: &str = "\u{FEFF}FD\u{FEFF}";            // Floating point decimal
    pub const LEADING_DOT: &str = "\u{FEFF}LD\u{FEFF}";          // Leading decimal (.625)
    pub const ABBREV_DOT: &str = "\u{FEFF}AD\u{FEFF}";           // Abbreviation dots (U.S.A.)
    pub const INITIAL_DOT: &str = "\u{FEFF}ID\u{FEFF}";          // Initial dots (J.)
    pub const TITLE_DOT: &str = "\u{FEFF}TD\u{FEFF}";            // Title dots (Dr. Mr.)
    pub const PAREN_END: &str = "\u{FEFF}PE\u{FEFF}";            // Sentence ender before paren
    pub const QUOTE_SINGLE_DOUBLE: &str = "\u{FEFF}QSD\u{FEFF}"; // '. "
    pub const QUOTE_CURLY_END: &str = "\u{FEFF}QCE\u{FEFF}";     // ."
    pub const QUOTE_DOUBLE_SINGLE: &str = "\u{FEFF}QDS\u{FEFF}"; // '" pattern
    pub const QUOTE_SINGLE: &str = "\u{FEFF}QS\u{FEFF}";         // .'
    pub const QUOTE_CURLY_CLOSE: &str = "\u{FEFF}QCC\u{FEFF}";   // ."
}

// Unicode curly quote characters
const CURLY_CLOSE_SINGLE: char = '\u{2019}'; // '
const CURLY_CLOSE_DOUBLE: char = '\u{201D}'; // "

// Pre-compiled regexes for pattern protection
// Matches "et al." optionally followed by another period
static COMPOSITE_ABBREV: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?P<comp>et al)\.(?:\.)?").unwrap());

static SUSPENSION_POINTS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\.{3}").unwrap());

static FLOAT_POINT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?P<number>[0-9]+)\.(?P<decimal>[0-9]+)").unwrap());

static LEADING_DECIMAL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\s\.(?P<nums>[0-9]+)").unwrap());

static ABBREVIATION: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:[A-Za-z]\.){2,}").unwrap());

static INITIALS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?P<init>[A-Z])(?P<point>\.)").unwrap());

static TITLES: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?P<title>[A-Z][a-z]{1,3})(\.)").unwrap());

static UNSTICK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?P<left>[^.?!]\.|!|\?)(?P<right>[^\s"'])"#).unwrap());

static BEFORE_PARENS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?P<bef>[.?!])\s?\)").unwrap());

static QUOTE_SINGLE_DOUBLE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"'(?P<quote>[.?!])\s?""#).unwrap());

// Curly quote patterns - built dynamically
static QUOTE_CURLY_SINGLE_DOUBLE: Lazy<Regex> =
    Lazy::new(|| {
        let pattern = format!(r"{}(?P<quote>[.?!])\s?{}", CURLY_CLOSE_SINGLE, CURLY_CLOSE_DOUBLE);
        Regex::new(&pattern).unwrap()
    });

static QUOTE_CURLY_END: Lazy<Regex> =
    Lazy::new(|| {
        let pattern = format!(r"(?P<quote>[.?!])\s?{}", CURLY_CLOSE_DOUBLE);
        Regex::new(&pattern).unwrap()
    });

static QUOTE_DOUBLE_SINGLE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?P<quote>[.?!])\s?'""#).unwrap());

static QUOTE_SINGLE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?P<quote>[.?!])\s?'").unwrap());

static QUOTE_CURLY_CLOSE: Lazy<Regex> =
    Lazy::new(|| {
        let pattern = format!(r"(?P<quote>[.?!])\s?{}", CURLY_CLOSE_DOUBLE);
        Regex::new(&pattern).unwrap()
    });

/// Splits text into sentences using regex-based boundary detection.
///
/// This function handles common edge cases including:
/// - Abbreviations (U.S.A., Dr., Mr., etc.)
/// - Initials (J. K. Rowling)
/// - Floating-point numbers (3.14, .625)
/// - Suspension points (...)
/// - Quoted sentence endings ("Hello." vs Hello.")
/// - Parenthetical sentences (This is a test.)
///
/// # Arguments
/// * `text` - The input text to segment
///
/// # Returns
/// A vector of sentence strings
///
/// # Example
/// ```
/// use proteus::segmentation::sentence::split_sentences;
///
/// let text = "Dr. Smith went to Washington. He arrived at 3.14 p.m.";
/// let sentences = split_sentences(text);
/// assert_eq!(sentences.len(), 2);
/// ```
pub fn split_sentences(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    // Step 1: Protect special patterns by replacing them with placeholders
    let mut protected = text.to_string();

    // Composite abbreviations (et al.)
    protected = COMPOSITE_ABBREV
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}", &caps["comp"], placeholder::COMPOSITE_DOT)
        })
        .into_owned();

    // Suspension points (...)
    protected = SUSPENSION_POINTS
        .replace_all(&protected, placeholder::SUSPENSION)
        .into_owned();

    // Floating-point numbers (3.14)
    protected = FLOAT_POINT
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}{}", &caps["number"], placeholder::FLOAT_DOT, &caps["decimal"])
        })
        .into_owned();

    // Leading decimals (.625)
    protected = LEADING_DECIMAL
        .replace_all(&protected, |caps: &Captures| {
            format!(" {}{}", placeholder::LEADING_DOT, &caps["nums"])
        })
        .into_owned();

    // Multi-letter abbreviations (U.S.A., N.A.S.A.)
    protected = ABBREVIATION
        .replace_all(&protected, |caps: &Captures| {
            caps[0].replace('.', placeholder::ABBREV_DOT)
        })
        .into_owned();

    // Single-letter initials (J.)
    protected = INITIALS
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}", &caps["init"], placeholder::INITIAL_DOT)
        })
        .into_owned();

    // Titles (Dr., Mr., Mrs., Ms., Jr., Sr., etc.)
    protected = TITLES
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}", &caps["title"], placeholder::TITLE_DOT)
        })
        .into_owned();

    // Unstick sentences that got concatenated (e.g., "word.Word" -> "word. Word")
    protected = UNSTICK
        .replace_all(&protected, "$left $right")
        .into_owned();

    // Protect sentence enders before closing parens
    protected = BEFORE_PARENS
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{})", placeholder::PAREN_END, &caps["bef"])
        })
        .into_owned();

    // Protect various quote patterns
    protected = QUOTE_SINGLE_DOUBLE
        .replace_all(&protected, |caps: &Captures| {
            format!("'{}{}", placeholder::QUOTE_SINGLE_DOUBLE, &caps["quote"])
        })
        .into_owned();

    protected = QUOTE_CURLY_SINGLE_DOUBLE
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}{}", CURLY_CLOSE_SINGLE, placeholder::QUOTE_SINGLE_DOUBLE, &caps["quote"])
        })
        .into_owned();

    protected = QUOTE_CURLY_END
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}", placeholder::QUOTE_CURLY_END, &caps["quote"])
        })
        .into_owned();

    protected = QUOTE_DOUBLE_SINGLE
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}", placeholder::QUOTE_DOUBLE_SINGLE, &caps["quote"])
        })
        .into_owned();

    protected = QUOTE_SINGLE
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}", placeholder::QUOTE_SINGLE, &caps["quote"])
        })
        .into_owned();

    protected = QUOTE_CURLY_CLOSE
        .replace_all(&protected, |caps: &Captures| {
            format!("{}{}", placeholder::QUOTE_CURLY_CLOSE, &caps["quote"])
        })
        .into_owned();

    // Step 2: Split on sentence-ending punctuation
    let mut sentences = split_on_enders(&protected);

    // Step 3: Repair the placeholders back to original text
    sentences = sentences
        .into_iter()
        .map(|s| repair_sentence(&s))
        .filter(|s| !s.is_empty())
        .collect();

    sentences
}

/// Splits text on sentence-ending punctuation (. ! ?)
fn split_on_enders(text: &str) -> Vec<String> {
    let mut result = Vec::new();

    // Split on ! first
    let exclaim_parts: Vec<&str> = text.split('!').collect();
    let mut after_exclaim: Vec<String> = Vec::new();

    for (i, part) in exclaim_parts.iter().enumerate() {
        if i < exclaim_parts.len() - 1 {
            after_exclaim.push(format!("{}!", part));
        } else {
            after_exclaim.push(part.to_string());
        }
    }

    // Split each part on ?
    let mut after_question: Vec<String> = Vec::new();
    for part in after_exclaim {
        let question_parts: Vec<&str> = part.split('?').collect();
        for (i, qpart) in question_parts.iter().enumerate() {
            if i < question_parts.len() - 1 {
                after_question.push(format!("{}?", qpart));
            } else {
                after_question.push(qpart.to_string());
            }
        }
    }

    // Split each part on .
    for part in after_question {
        let period_parts: Vec<&str> = part.split('.').collect();
        for (i, ppart) in period_parts.iter().enumerate() {
            if i < period_parts.len() - 1 {
                result.push(format!("{}.", ppart));
            } else {
                result.push(ppart.to_string());
            }
        }
    }

    result
}

/// Repairs placeholders back to original text
fn repair_sentence(s: &str) -> String {
    let mut result = s.trim().to_string();

    // Basic placeholder replacements
    result = result.replace(placeholder::COMPOSITE_DOT, ".");
    result = result.replace(placeholder::SUSPENSION, "...");
    result = result.replace(placeholder::FLOAT_DOT, ".");
    result = result.replace(placeholder::LEADING_DOT, ".");
    result = result.replace(placeholder::ABBREV_DOT, ".");
    result = result.replace(placeholder::INITIAL_DOT, ".");
    result = result.replace(placeholder::TITLE_DOT, ".");

    // Repair parenthetical endings
    result = result.replace(&format!("{}.", placeholder::PAREN_END), ".)");
    result = result.replace(&format!("{}!", placeholder::PAREN_END), "!)");
    result = result.replace(&format!("{}?", placeholder::PAREN_END), "?)");

    // Repair quote patterns
    result = repair_quote_pattern(&result, placeholder::QUOTE_SINGLE_DOUBLE, |p| {
        format!("'{}\"", p)
    });
    result = repair_quote_pattern(&result, placeholder::QUOTE_CURLY_END, |p| {
        format!("{}{}", p, CURLY_CLOSE_DOUBLE)
    });
    result = repair_quote_pattern(&result, placeholder::QUOTE_DOUBLE_SINGLE, |p| {
        format!("{}\"'", p)
    });
    result = repair_quote_pattern(&result, placeholder::QUOTE_SINGLE, |p| {
        format!("{}'", p)
    });
    result = repair_quote_pattern(&result, placeholder::QUOTE_CURLY_CLOSE, |p| {
        format!("{}{}", p, CURLY_CLOSE_DOUBLE)
    });

    result
}

/// Helper to repair quote placeholders
fn repair_quote_pattern<F>(text: &str, placeholder: &str, replacement: F) -> String
where
    F: Fn(&str) -> String,
{
    let mut result = text.to_string();

    for punct in ['.', '!', '?'] {
        let pattern = format!("{}{}", placeholder, punct);
        if result.contains(&pattern) {
            result = result.replace(&pattern, &replacement(&punct.to_string()));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sentences() {
        let text = "Hello world. This is a test.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "This is a test.");
    }

    #[test]
    fn test_abbreviations() {
        let text = "Dr. Smith went to Washington.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "Dr. Smith went to Washington.");
    }

    #[test]
    fn test_multi_letter_abbreviations() {
        let text = "People in the U.S.A. love freedom.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn test_initials() {
        let text = "J. K. Rowling wrote Harry Potter. It was successful.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_floating_point() {
        let text = "The value was 3.14159. Then it changed.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("3.14159"));
    }

    #[test]
    fn test_leading_decimal() {
        let text = "The result was .625. That is correct.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains(".625"));
    }

    #[test]
    fn test_suspension_points() {
        let text = "And then... it happened. Something amazing.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("..."));
    }

    #[test]
    fn test_question_and_exclamation() {
        let text = "Is this working? Yes it is! Great.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].ends_with('?'));
        assert!(sentences[1].ends_with('!'));
        assert!(sentences[2].ends_with('.'));
    }

    #[test]
    fn test_parenthetical_sentence() {
        let text = "(This is a sentence.) And this is another.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_complex_text() {
        let text = "For years, people in the U.A.E.R. have accepted murky air. \
            But public dissent has been growing! \
            In July alone, two demonstrations turned violent after about 1.5 minutes... \
            The man found the result to be .625. \
            (This is another sentence in parens.) \
            This is the last sentence.";

        let sentences = split_sentences(text);

        assert!(sentences.len() >= 5, "Expected at least 5 sentences, got {}: {:?}", sentences.len(), sentences);
        assert!(sentences[0].contains("U.A.E.R."), "First sentence should contain U.A.E.R.");
        assert!(sentences.iter().any(|s| s.contains("1.5 minutes")), "Should have sentence with 1.5");
        assert!(sentences.iter().any(|s| s.contains(".625")), "Should have sentence with .625");
    }

    #[test]
    fn test_empty_input() {
        let sentences = split_sentences("");
        assert!(sentences.is_empty());
    }

    #[test]
    fn test_no_sentences() {
        let text = "No ending punctuation here";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "No ending punctuation here");
    }

    #[test]
    fn test_et_al() {
        let text = "According to Smith et al. the results were clear. This was expected.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2, "Expected 2 sentences, got: {:?}", sentences);
    }

    #[test]
    fn test_dr_c_jeung() {
        // Test title + initial combination
        let text = "Dr. C. Jeung studied the data. It was conclusive.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("Dr. C. Jeung"));
    }

    #[test]
    fn test_quoted_sentences() {
        let text = r#"He said "Hello." Then he left."#;
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
    }
}
