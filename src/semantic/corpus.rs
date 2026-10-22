//! Offset-preserving dependency annotations supplied by an identified parser.

use super::invalid;
use crate::Result;
use serde::{Deserialize, Serialize};

/// Parser and input identities carried with every document.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ParserProvenance {
    /// Parser implementation name.
    pub name: String,
    /// Parser package version.
    pub version: String,
    /// Model identity and version.
    pub model: String,
    /// Hash of the parser model directory manifest.
    pub model_sha256: String,
    /// Hash of the original unmodified input bytes.
    pub source_sha256: String,
    /// Numerical execution backend; older preparation artifacts leave it unknown.
    #[serde(default = "unknown_backend")]
    pub backend: String,
}

fn unknown_backend() -> String {
    "unknown".into()
}

/// All offsets are half-open UTF-8 byte ranges into the enclosing document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedToken {
    /// Inclusive UTF-8 byte offset into the document text.
    pub start: usize,
    /// Exclusive UTF-8 byte offset into the document text.
    pub end: usize,
    /// Parser-supplied lemma; original spelling remains available by byte range.
    pub lemma: String,
    /// Parser universal part-of-speech tag.
    pub pos: String,
    /// Dependency label in the declared parser dialect.
    pub dependency: String,
    /// Zero-based document token index of the dependency head; roots head themselves.
    pub head: usize,
    /// Zero-based sentence identity within this document.
    pub sentence: usize,
    /// Parser-supplied morphology, for example Tense=Past or VerbForm=Fin.
    #[serde(default)]
    pub morphology: Vec<String>,
}

/// Original document text and validated dependency annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedDocument {
    /// Unique document identity within the completed annotation stream.
    pub id: String,
    /// Document start in UTF-8 bytes relative to the original source input.
    #[serde(default)]
    pub source_start: usize,
    /// Unmodified document text used by every retained byte range.
    pub text: String,
    /// Identity of the parser and the source input.
    pub parser: ParserProvenance,
    /// Ordered, nonoverlapping source tokens and dependency heads.
    pub tokens: Vec<AnnotatedToken>,
}

impl AnnotatedDocument {
    /// Validate byte ranges, head references and acyclic sentence-local dependency forests.
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty()
            || self.text.is_empty()
            || self.tokens.is_empty()
            || self.parser.name.is_empty()
            || self.parser.version.is_empty()
            || self.parser.model.is_empty()
            || !super::is_sha256(&self.parser.model_sha256)
            || !super::is_sha256(&self.parser.source_sha256)
        {
            return Err(invalid(
                "Missing document text, tokens or parser provenance",
            ));
        }
        let mut previous_end = 0;
        let mut previous_sentence = 0;
        for token in &self.tokens {
            if token.start < previous_end
                || token.start >= token.end
                || self.text.get(token.start..token.end).is_none()
                || token.head >= self.tokens.len()
                || token.sentence < previous_sentence
                || self.tokens[token.head].sentence != token.sentence
                || token.lemma.is_empty()
                || token.pos.is_empty()
                || token.dependency.is_empty()
            {
                return Err(invalid("Invalid token span, sentence or dependency head"));
            }
            previous_end = token.end;
            previous_sentence = token.sentence;
        }
        // Each head chain must end at a self-headed root. Iterative coloring
        // avoids recursion and validates the complete forest in O(tokens).
        let mut colors = vec![0_u8; self.tokens.len()];
        for start in 0..self.tokens.len() {
            if colors[start] != 0 {
                continue;
            }
            let mut path = Vec::new();
            let mut current = start;
            while colors[current] == 0 {
                colors[current] = 1;
                path.push(current);
                if self.tokens[current].head == current {
                    break;
                }
                current = self.tokens[current].head;
            }
            if colors[current] == 1 && self.tokens[current].head != current {
                return Err(invalid("Cycle in dependency annotations"));
            }
            for position in path {
                colors[position] = 2;
            }
        }
        Ok(())
    }
}

/// Read a completed JSONL corpus. The required final digest rejects truncated
/// successful prefixes, including failures in an upstream preparation process.
pub fn read_annotated_corpus(reader: impl std::io::BufRead) -> Result<Vec<AnnotatedDocument>> {
    use sha2::{Digest, Sha256};

    let mut documents = Vec::new();
    let mut identities = std::collections::BTreeSet::new();
    let mut digest = Sha256::new();
    let mut complete = false;
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if complete {
            return Err(invalid("Data follows the annotation completion record"));
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .map_err(|error| invalid(&format!("Annotation line {}: {error}", line_number + 1)))?;
        if let Some(count) = value.get("complete") {
            if count.as_u64() != Some(documents.len() as u64)
                || value.get("schema").and_then(serde_json::Value::as_u64) != Some(1)
                || value
                    .get("documents_sha256")
                    .and_then(serde_json::Value::as_str)
                    != Some(format!("{:x}", digest.clone().finalize()).as_str())
            {
                return Err(invalid("Annotation completion count or digest mismatch"));
            }
            complete = true;
            continue;
        }
        let document: AnnotatedDocument = serde_json::from_str(&line)
            .map_err(|error| invalid(&format!("Annotation line {}: {error}", line_number + 1)))?;
        document.validate()?;
        if !identities.insert(document.id.clone()) {
            return Err(invalid("Duplicate annotated document identity"));
        }
        documents.push(document);
        digest.update(line.as_bytes());
        digest.update(b"\n");
    }
    if !complete {
        return Err(invalid("Missing annotation completion record"));
    }
    if documents.is_empty() {
        return Err(crate::ProteusError::EmptyInput(
            "No annotated documents".into(),
        ));
    }
    Ok(documents)
}
