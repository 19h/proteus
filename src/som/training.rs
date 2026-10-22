//! Deterministic online Kohonen training. All assignments are made on the final map.
//!
//! Each update uses a freshly computed, exhaustive Euclidean BMU and the full
//! Gaussian neighbourhood. Parallelism is across independent neuron updates,
//! never across stale BMUs. See `docs/algorithm.md` for the numerical contract.

use crate::config::SomConfig;
use crate::error::{ProteusError, Result};
use crate::som::simd::{add_vectors_f32, find_bmu_f32, normalize_f32, update_weights_f32};
use crate::som::Som;
use indicatif::ProgressBar;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};

/// Dimension of the legacy random-indexing adapter, not a learned embedding model.
pub const DEFAULT_EMBEDDING_DIM: usize = 100;

/// A labelled training vector for the low-level SOM interface.
#[derive(Debug, Clone)]
pub struct TrainingContext {
    /// Label associated with this sample.
    pub center: String,
    /// Finite input coordinates.
    pub embedding: Vec<f64>,
}

impl TrainingContext {
    /// Creates a labelled vector.
    pub fn new(center: String, embedding: Vec<f64>) -> Self {
        Self { center, embedding }
    }
}

/// Compatibility adapter implementing seeded random indexing of co-occurrences.
/// This is not skip-gram and is not used by the corpus training pipeline.
pub struct WordEmbeddings {
    embeddings: HashMap<String, Vec<f32>>,
    dim: usize,
}

impl WordEmbeddings {
    /// Accumulates context-word random vectors in canonical lexical order.
    pub fn from_contexts(
        contexts: &[(String, Vec<String>)],
        dim: usize,
        seed: Option<u64>,
    ) -> Self {
        Self::from_contexts_with_progress(contexts, dim, seed, None)
    }

    /// Random indexing with progress reporting. The same seed and data produce
    /// identical vectors, independently of hash-map and Rayon iteration order.
    pub fn from_contexts_with_progress(
        contexts: &[(String, Vec<String>)],
        dim: usize,
        seed: Option<u64>,
        progress: Option<&ProgressBar>,
    ) -> Self {
        let mut rng = rng(seed);
        let words: BTreeSet<&str> = contexts
            .iter()
            .flat_map(|(w, c)| std::iter::once(w.as_str()).chain(c.iter().map(String::as_str)))
            .collect();
        let projections: HashMap<&str, Vec<f32>> = words
            .into_iter()
            .map(|w| (w, (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()))
            .collect();
        // Integer counts first: floating-point reduction order cannot depend on threads.
        let mut counts: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
        for (w, ctx) in contexts {
            let row = counts.entry(w).or_default();
            for c in ctx {
                *row.entry(c).or_default() += 1;
            }
        }
        let embeddings = counts
            .into_iter()
            .map(|(w, row)| {
                let mut row: Vec<_> = row.into_iter().collect();
                row.sort_unstable_by_key(|&(c, _)| c);
                let mut vector = vec![0.0; dim];
                for (c, count) in row {
                    for (out, value) in vector.iter_mut().zip(&projections[c]) {
                        *out += *value * count as f32;
                    }
                }
                normalize_f32(&mut vector);
                (w.to_owned(), vector)
            })
            .collect();
        if let Some(pb) = progress {
            pb.set_position(contexts.len() as u64);
        }
        Self { embeddings, dim }
    }

    /// Looks up a random-indexing vector.
    pub fn get(&self, word: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(word)
    }
    /// Number of coordinates.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Vocabulary size.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }
    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

fn rng(seed: Option<u64>) -> ChaCha8Rng {
    seed.map(ChaCha8Rng::seed_from_u64)
        .unwrap_or_else(ChaCha8Rng::from_entropy)
}

/// Validates shape and schedules before allocating or training a map.
pub fn validate_config(config: &SomConfig) -> Result<()> {
    let invalid = |message: &str| ProteusError::Config(message.to_owned());
    if config.dimension == 0
        || config
            .dimension
            .checked_mul(config.dimension)
            .filter(|&n| n <= u32::MAX as usize)
            .is_none()
    {
        return Err(invalid(
            "SOM dimension must be positive and its square fit in u32",
        ));
    }
    if config.weight_dimension == 0 || config.iterations == 0 {
        return Err(invalid("Weight dimension and iterations must be positive"));
    }
    for value in [config.initial_learning_rate, config.final_learning_rate] {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(invalid("Learning rates must be finite and in (0, 1]"));
        }
    }
    for value in [config.initial_radius, config.final_radius] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid("Neighbourhood radii must be finite and positive"));
        }
    }
    Ok(())
}

/// Online SOM trainer. Seeded sample order and exact BMUs are shared by all APIs.
pub struct SomTrainer {
    config: SomConfig,
    rng: ChaCha8Rng,
}

impl SomTrainer {
    /// Creates a trainer. Training methods validate the configuration.
    pub fn new(config: SomConfig) -> Self {
        let rng = rng(config.seed);
        Self { config, rng }
    }

    fn schedule(&self, iteration: usize, initial: f64, final_value: f64) -> f64 {
        let t = if self.config.iterations <= 1 {
            0.0
        } else {
            iteration.min(self.config.iterations - 1) as f64 / (self.config.iterations - 1) as f64
        };
        // Log-space interpolation also avoids overflow in final_value / initial.
        ((1.0 - t) * initial.ln() + t * final_value.ln()).exp()
    }

    /// Exponential learning rate; first and last updates reach the endpoints.
    pub fn learning_rate(&self, iteration: usize) -> f64 {
        self.schedule(
            iteration,
            self.config.initial_learning_rate,
            self.config.final_learning_rate,
        )
    }
    /// Exponential Gaussian radius, in grid cells.
    pub fn radius(&self, iteration: usize) -> f64 {
        self.schedule(
            iteration,
            self.config.initial_radius,
            self.config.final_radius,
        )
    }

    /// Train finite f32 vectors. Returns one final-map BMU per input vector.
    /// Exactly `iterations` online updates are executed, followed by a frozen-map
    /// assignment pass over *every* input, including samples never drawn in training.
    pub fn train_vectors<F>(
        &mut self,
        som: &mut Som,
        inputs: &[Vec<f32>],
        progress: F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        validate_config(&self.config)?;
        if inputs.is_empty() {
            return Err(ProteusError::Training("No training vectors".into()));
        }
        let d = self.config.weight_dimension;
        let n = self.config.total_neurons();
        if som.dimension != self.config.dimension
            || som.weight_dim != d
            || som.neurons.len() != n
            || som.toroidal != self.config.toroidal
        {
            return Err(ProteusError::Som(
                "Map and trainer configurations differ".into(),
            ));
        }
        if inputs
            .iter()
            .any(|v| v.len() != d || v.iter().any(|x| !x.is_finite()))
            || som
                .neurons
                .iter()
                .any(|v| v.weights.len() != d || v.weights.iter().any(|&x| !(x as f32).is_finite()))
        {
            return Err(ProteusError::Som(
                "Input or weight vector has invalid dimension/nonfinite coordinates".into(),
            ));
        }
        // Bound squared distances away from overflow even for opposite-sign vectors.
        let bound = (f32::MAX / (8.0 * d as f32)).sqrt();
        if inputs.iter().flatten().any(|x| x.abs() > bound)
            || som
                .neurons
                .iter()
                .flat_map(|n| &n.weights)
                .any(|w| w.abs() > bound as f64)
        {
            return Err(ProteusError::Som(
                "Vector magnitude would overflow squared f32 distance".into(),
            ));
        }
        let threads = self.config.num_threads;
        let run = || self.train_vectors_inner(som, inputs, progress);
        if threads == 0 {
            run()
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| ProteusError::Training(e.to_string()))?
                .install(run)
        }
    }

    fn train_vectors_inner<F>(
        &mut self,
        som: &mut Som,
        inputs: &[Vec<f32>],
        progress: F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let d = som.weight_dim;
        let dim = som.dimension;
        let n = som.neurons.len();
        let mut weights: Vec<f32> = som
            .neurons
            .iter()
            .flat_map(|v| v.weights.iter().map(|&w| w as f32))
            .collect();
        let mut order: Vec<usize> = (0..inputs.len()).collect();
        let mut axis_gaussian = vec![0.0f64; dim];
        let mut gaussian = vec![0.0f32; n];
        for t in 0..self.config.iterations {
            if t % order.len() == 0 {
                order.shuffle(&mut self.rng);
            }
            let input = &inputs[order[t % order.len()]];
            let bmu = find_bmu_f32(&weights, input, n, d);
            let (br, bc) = (bmu / dim, bmu % dim);
            let lr = self.learning_rate(t);
            let radius = self.radius(t);
            // exp(-(dr^2+dc^2)/(2*sigma^2)) separates into two 1-D
            // Gaussians. This evaluates O(dim), not O(dim^2), exponentials.
            for (distance, h) in axis_gaussian.iter_mut().enumerate() {
                *h = (-0.5 * (distance as f64 / radius).powi(2)).exp();
            }
            for (dr, row) in gaussian.chunks_exact_mut(dim).enumerate() {
                for (dc, h) in row.iter_mut().enumerate() {
                    *h = (lr * axis_gaussian[dr] * axis_gaussian[dc]) as f32;
                }
            }
            let update = |(i, w): (usize, &mut [f32])| {
                let mut dr = br.abs_diff(i / dim);
                let mut dc = bc.abs_diff(i % dim);
                if som.toroidal {
                    dr = dr.min(dim - dr);
                    dc = dc.min(dim - dc);
                }
                update_weights_f32(w, input, gaussian[dr * dim + dc]);
            };
            // Per-neuron arithmetic is identical with either scheduling strategy.
            if n * d >= 262_144 {
                weights.par_chunks_mut(d).enumerate().for_each(update);
            } else {
                weights.chunks_mut(d).enumerate().for_each(update);
            }
            if t % 100 == 0 || t + 1 == self.config.iterations {
                progress(t + 1, self.config.iterations);
            }
        }
        let assignments = inputs
            .par_iter()
            .map(|v| find_bmu_f32(&weights, v, n, d))
            .collect();
        for (neuron, w) in som.neurons.iter_mut().zip(weights.chunks_exact(d)) {
            for (dst, &src) in neuron.weights.iter_mut().zip(w) {
                *dst = src as f64;
            }
        }
        Ok(assignments)
    }

    /// Legacy random-indexed context adapter, using the same online trainer.
    pub fn train_fast(
        &mut self,
        som: &mut Som,
        embeddings: &WordEmbeddings,
        contexts: &[(String, Vec<String>)],
    ) -> Result<HashMap<String, Vec<usize>>> {
        self.train_fast_with_progress(som, embeddings, contexts, |_, _, _, _| {})
    }

    /// Legacy adapter with a live update-count callback.
    pub fn train_fast_with_progress<F>(
        &mut self,
        som: &mut Som,
        embeddings: &WordEmbeddings,
        contexts: &[(String, Vec<String>)],
        callback: F,
    ) -> Result<HashMap<String, Vec<usize>>>
    where
        F: Fn(usize, usize, usize, &str) + Send + Sync,
    {
        let mut labels = Vec::new();
        let mut inputs = Vec::new();
        for (word, context) in contexts {
            let mut v = vec![0.0; embeddings.dim()];
            for w in context {
                if let Some(e) = embeddings.get(w) {
                    add_vectors_f32(&mut v, e);
                }
            }
            if v.iter().all(|&x| x == 0.0) {
                continue;
            }
            normalize_f32(&mut v);
            labels.push(word.clone());
            inputs.push(v);
        }
        let bmus = self.train_vectors(som, &inputs, |i, n| callback(i, n, 0, "online SOM"))?;
        Ok(group_assignments(labels.into_iter().zip(bmus)))
    }

    /// Trains labelled vectors and groups their frozen-map assignments by label.
    pub fn train(
        &mut self,
        som: &mut Som,
        contexts: &[TrainingContext],
    ) -> Result<HashMap<String, Vec<usize>>> {
        let inputs: Vec<Vec<f32>> = contexts
            .iter()
            .map(|c| c.embedding.iter().map(|&x| x as f32).collect())
            .collect();
        let bmus = self.train_vectors(som, &inputs, |_, _| {})?;
        Ok(group_assignments(
            contexts.iter().map(|c| c.center.clone()).zip(bmus),
        ))
    }

    /// Keeps the most frequent final-map positions, with a stable position tie-break.
    pub fn compute_fingerprints(
        mapping: &HashMap<String, Vec<usize>>,
        target_bits: usize,
    ) -> HashMap<String, Vec<usize>> {
        mapping
            .iter()
            .map(|(word, bmus)| {
                let mut counts = HashMap::<usize, usize>::new();
                for &bmu in bmus {
                    *counts.entry(bmu).or_default() += 1;
                }
                let mut counts: Vec<_> = counts.into_iter().collect();
                counts.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                (
                    word.clone(),
                    counts.into_iter().take(target_bits).map(|x| x.0).collect(),
                )
            })
            .collect()
    }
}

fn group_assignments(iter: impl Iterator<Item = (String, usize)>) -> HashMap<String, Vec<usize>> {
    let mut result = HashMap::<String, Vec<usize>>::new();
    for (word, bmu) in iter {
        result.entry(word).or_default().push(bmu);
    }
    result
}
