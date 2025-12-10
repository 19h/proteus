//! HuggingFace Hub integration for model downloading

use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};

use crate::wtpsplit::error::Error;
use crate::wtpsplit::Result;

/// Default hub prefix for SaT models
pub const SAT_HUB_PREFIX: &str = "segment-any-text";

/// Default hub prefix for WtP models
pub const WTP_HUB_PREFIX: &str = "benjamin";

/// Model files that need to be downloaded
pub struct ModelFiles {
    /// Path to the config.json file
    pub config_path: PathBuf,
    /// Path to the ONNX model file
    pub onnx_path: PathBuf,
    /// Optional path to the tokenizer.json file
    pub tokenizer_path: Option<PathBuf>,
}

/// Download model files from HuggingFace Hub
///
/// # Arguments
/// * `model_name` - Model name (e.g., "sat-3l-sm")
/// * `hub_prefix` - Hub username/organization prefix
/// * `use_optimized` - Whether to use the optimized ONNX model
///
/// # Returns
/// Paths to the downloaded model files
pub fn download_model(
    model_name: &str,
    hub_prefix: Option<&str>,
    use_optimized: bool,
) -> Result<ModelFiles> {
    let repo_id = if let Some(prefix) = hub_prefix {
        format!("{}/{}", prefix, model_name)
    } else {
        model_name.to_string()
    };

    log::info!("Downloading model from {}", repo_id);

    let api = Api::new().map_err(|e| Error::HubDownload(e.to_string()))?;
    let repo = api.repo(Repo::new(repo_id.clone(), RepoType::Model));

    // Download config.json
    let config_path = repo
        .get("config.json")
        .map_err(|e| Error::HubDownload(format!("Failed to download config.json: {}", e)))?;

    // Download ONNX model
    let onnx_filename = if use_optimized {
        "model_optimized.onnx"
    } else {
        "model.onnx"
    };

    let onnx_path = repo
        .get(onnx_filename)
        .map_err(|e| Error::HubDownload(format!("Failed to download {}: {}", onnx_filename, e)))?;

    // Try to download tokenizer files (may not exist for all models)
    let tokenizer_path = repo.get("tokenizer.json").ok();

    Ok(ModelFiles {
        config_path,
        onnx_path,
        tokenizer_path,
    })
}

/// Check if a path is a local directory
pub fn is_local_path(path: &str) -> bool {
    Path::new(path).is_dir()
}

/// Get model files from a local directory
pub fn get_local_model_files(model_dir: &Path, use_optimized: bool) -> Result<ModelFiles> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "config.json not found in {}",
            model_dir.display()
        )));
    }

    let onnx_filename = if use_optimized {
        "model_optimized.onnx"
    } else {
        "model.onnx"
    };

    let onnx_path = model_dir.join(onnx_filename);
    if !onnx_path.exists() {
        // Try the other variant
        let alt_filename = if use_optimized {
            "model.onnx"
        } else {
            "model_optimized.onnx"
        };
        let alt_path = model_dir.join(alt_filename);
        if alt_path.exists() {
            return Ok(ModelFiles {
                config_path,
                onnx_path: alt_path,
                tokenizer_path: model_dir.join("tokenizer.json").exists().then(|| model_dir.join("tokenizer.json")),
            });
        }
        return Err(Error::ModelNotFound(format!(
            "ONNX model not found in {}",
            model_dir.display()
        )));
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer_path = if tokenizer_path.exists() {
        Some(tokenizer_path)
    } else {
        None
    };

    Ok(ModelFiles {
        config_path,
        onnx_path,
        tokenizer_path,
    })
}

/// Download XLM-RoBERTa tokenizer from HuggingFace
pub fn download_xlm_roberta_tokenizer() -> Result<PathBuf> {
    let api = Api::new().map_err(|e| Error::HubDownload(e.to_string()))?;
    let repo = api.repo(Repo::new(
        "FacebookAI/xlm-roberta-base".to_string(),
        RepoType::Model,
    ));

    repo.get("tokenizer.json")
        .map_err(|e| Error::HubDownload(format!("Failed to download XLM-RoBERTa tokenizer: {}", e)))
}
