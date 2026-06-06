//! # Entropy Conservation
//!
//! Core mathematical framework implementing the Conservation of Verification Entropy —
//! the meta-law discovered across all PLATO/SuperInstance experiments.
//!
//! **Principle:** Every closed system of verification (tests, proofs, type checks)
//! has a conserved quantity H — the verification entropy.
//!
//! H = -Σ pᵢ log pᵢ, where pᵢ is the probability that verification path i is exercised.
//!
//! **Conservation law:** dH/dt ≤ 0 (entropy never increases in a well-structured system).

pub mod conservation;
pub mod entropy;
pub mod flow;
pub mod gradient;
pub mod hodge;
pub mod persistence;
pub mod spectrum;

pub use conservation::{ConservationReport, EntropyViolation};
pub use entropy::{VerificationEntropy};
pub use flow::{EntropyFlow, ModuleId};
pub use gradient::{GradientSuggestion, GradientDescent};
pub use hodge::HodgeDecomposition;
pub use persistence::PersistenceDiagram;
pub use spectrum::SpectralAnalysis;

/// Error type for entropy-conservation operations.
#[derive(Debug, thiserror::Error)]
pub enum EntropyError {
    #[error("invalid probability distribution: probabilities must sum to 1.0, got {0}")]
    InvalidDistribution(f64),
    #[error("negative probability encountered: {0}")]
    NegativeProbability(f64),
    #[error("empty probability vector")]
    EmptyVector,
    #[error("singular matrix: {0}")]
    SingularMatrix(String),
    #[error("invalid Rényi order α: must be positive and ≠ 1 for Rényi, got {0}")]
    InvalidRenyiOrder(f64),
}
