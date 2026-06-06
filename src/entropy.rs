//! Shannon entropy, Rényi entropy (order α), Tsallis entropy, and
//! verification-entropy calculation from test-coverage vectors.

use std::collections::BTreeMap;

use crate::EntropyError;

/// Wrapper to use f64 as a map key (total ordering with NaN handling).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct OrderedFloat(pub f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        if self.0.is_nan() && other.0.is_nan() { true } else { self.0 == other.0 }
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or_else(|| {
            if self.0.is_nan() && !other.0.is_nan() { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Less }
        })
    }
}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.0.is_nan() {
            0x7ff8000000000000u64.hash(state);
        } else {
            self.0.to_bits().hash(state);
        }
    }
}

/// Core entropy measurement for a verification system.
///
/// Combines three entropy families:
/// - **Shannon** — the canonical H = -Σ pᵢ log pᵢ
/// - **Rényi** — generalised entropy H_α = (1/(1-α)) log Σ pᵢ^α
/// - **Tsallis** — S_q = (1/(q-1)) (1 - Σ pᵢ^q)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationEntropy {
    /// Shannon entropy H = -Σ pᵢ log₂ pᵢ
    pub shannon: f64,
    /// Rényi entropy keyed by order α
    pub renyi: BTreeMap<OrderedFloat, f64>,
    /// Tsallis entropy (q = 2 by default, can be recomputed)
    pub tsallis: f64,
}

impl VerificationEntropy {
    /// Compute verification entropy from a probability distribution.
    ///
    /// `probs` need not be normalised — we normalise internally.
    /// Uses base-2 logarithm (bits).
    pub fn from_probabilities(probs: &[f64]) -> Result<Self, EntropyError> {
        if probs.is_empty() {
            return Err(EntropyError::EmptyVector);
        }
        for &p in probs {
            if p < 0.0 {
                return Err(EntropyError::NegativeProbability(p));
            }
        }

        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            return Err(EntropyError::InvalidDistribution(total));
        }

        let p: Vec<f64> = probs.iter().map(|&x| x / total).collect();

        // Shannon
        let shannon = shannon_entropy(&p);

        // Rényi at several orders
        let mut renyi = BTreeMap::new();
        for &alpha in &[0.5, 1.5, 2.0, 3.0, f64::INFINITY] {
            if let Ok(h) = renyi_entropy(&p, alpha) {
                renyi.insert(OrderedFloat(alpha), h);
            }
        }

        // Tsallis (q = 2)
        let tsallis = tsallis_entropy(&p, 2.0);

        Ok(Self {
            shannon,
            renyi,
            tsallis,
        })
    }

    /// Convenience: compute from raw hit counts (e.g. per-path execution counts).
    pub fn from_counts(counts: &[u64]) -> Result<Self, EntropyError> {
        let probs: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        Self::from_probabilities(&probs)
    }

    /// Maximum possible Shannon entropy for n equiprobable paths.
    pub fn max_shannon(n: usize) -> f64 {
        if n == 0 { 0.0 } else { (n as f64).log2() }
    }

    /// Normalised Shannon entropy ∈ [0, 1].
    pub fn normalised(&self) -> f64 {
        let n = self.renyi.len().max(1);
        let max = Self::max_shannon(n);
        if max == 0.0 { 0.0 } else { self.shannon / max }
    }
}

/// Shannon entropy H = -Σ pᵢ log₂ pᵢ  (base-2, bits).
pub fn shannon_entropy(p: &[f64]) -> f64 {
    p.iter()
        .filter(|pi| **pi > 0.0)
        .map(|pi| -pi * pi.log2())
        .sum()
}

/// Rényi entropy of order α: H_α = (1/(1-α)) log₂(Σ pᵢ^α).
///
/// α → 1 limit recovers Shannon entropy.
/// α = 0  → Hartley (max) entropy log₂ n.
/// α = 2  → collision entropy.
/// α → ∞  → min-entropy.
pub fn renyi_entropy(p: &[f64], alpha: f64) -> Result<f64, EntropyError> {
    if alpha <= 0.0 {
        return Err(EntropyError::InvalidRenyiOrder(alpha));
    }
    if (alpha - 1.0).abs() < 1e-12 {
        // α ≈ 1 → Shannon
        return Ok(shannon_entropy(p));
    }
    if alpha == f64::INFINITY {
        return Ok(min_entropy(p));
    }
    let sum: f64 = p.iter().filter(|pi| **pi > 0.0).map(|pi| pi.powf(alpha)).sum();
    if sum <= 0.0 {
        return Ok(0.0);
    }
    Ok((1.0 / (1.0 - alpha)) * sum.log2())
}

/// Min-entropy: H_∞ = -log₂ max(pᵢ).
pub fn min_entropy(p: &[f64]) -> f64 {
    let max_p = p.iter().cloned().fold(0.0_f64, f64::max);
    if max_p <= 0.0 { 0.0 } else { -max_p.log2() }
}

/// Tsallis entropy S_q = (1/(q-1))(1 - Σ pᵢ^q).
pub fn tsallis_entropy(p: &[f64], q: f64) -> f64 {
    if (q - 1.0).abs() < 1e-12 {
        return shannon_entropy(p);
    }
    let sum: f64 = p.iter().filter(|pi| **pi > 0.0).map(|pi| pi.powf(q)).sum();
    (1.0 / (q - 1.0)) * (1.0 - sum)
}

/// Kullback-Leibler divergence D_KL(P ‖ Q).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|(pi, _)| **pi > 0.0)
        .map(|(pi, qi)| {
            let qi_safe = if *qi > 0.0 { *qi } else { 1e-300 };
            pi * (pi / qi_safe).log2()
        })
        .sum()
}

/// Cross-entropy H(P, Q) = -Σ pᵢ log₂ qᵢ.
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|(pi, _)| **pi > 0.0)
        .map(|(pi, qi)| {
            let qi_safe = if *qi > 0.0 { *qi } else { 1e-300 };
            -pi * qi_safe.log2()
        })
        .sum()
}

/// Jensen-Shannon divergence (symmetric, bounded).
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a, b)| 0.5 * (a + b)).collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_non_negative_uniform() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&p);
        assert!(h >= 0.0, "Shannon entropy must be non-negative");
    }

    #[test]
    fn entropy_non_negative_degenerate() {
        let p = vec![1.0, 0.0, 0.0];
        let h = shannon_entropy(&p);
        assert!(h >= 0.0);
        assert!(h < 1e-12, "deterministic distribution has ~0 entropy");
    }

    #[test]
    fn uniform_is_maximum_entropy() {
        let p = vec![0.25; 4];
        let h = shannon_entropy(&p);
        assert!((h - 2.0).abs() < 1e-12, "4 equiprobable paths → 2 bits");
    }

    #[test]
    fn verification_entropy_from_counts() {
        let ve = VerificationEntropy::from_counts(&[10, 20, 30, 40]).unwrap();
        assert!(ve.shannon >= 0.0);
        assert!(ve.shannon < 2.0, "non-uniform should be < max");
    }

    #[test]
    fn renyi_converges_to_shannon_as_alpha_1() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let shannon = shannon_entropy(&p);
        let renyi_near1 = renyi_entropy(&p, 1.0 + 1e-10).unwrap();
        assert!(
            (shannon - renyi_near1).abs() < 1e-6,
            "Rényi(α→1) should ≈ Shannon: shannon={shannon}, renyi={renyi_near1}"
        );
    }

    #[test]
    fn renyi_monotone_decreasing_in_alpha() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let h05 = renyi_entropy(&p, 0.5).unwrap();
        let h1 = shannon_entropy(&p);
        let h2 = renyi_entropy(&p, 2.0).unwrap();
        let h3 = renyi_entropy(&p, 3.0).unwrap();
        assert!(h05 >= h1 - 1e-12, "Rényi(0.5) ≥ Shannon");
        assert!(h1 >= h2 - 1e-12, "Shannon ≥ Rényi(2)");
        assert!(h2 >= h3 - 1e-12, "Rényi(2) ≥ Rényi(3)");
    }

    #[test]
    fn tsallis_non_negative() {
        let p = vec![0.3, 0.7];
        let t = tsallis_entropy(&p, 2.0);
        assert!(t >= 0.0, "Tsallis entropy must be non-negative for q>1");
    }

    #[test]
    fn kl_divergence_non_negative() {
        let p = vec![0.3, 0.7];
        let q = vec![0.5, 0.5];
        let d = kl_divergence(&p, &q);
        assert!(d >= -1e-12, "KL divergence is non-negative");
    }

    #[test]
    fn kl_divergence_zero_for_same() {
        let p = vec![0.3, 0.7];
        assert!((kl_divergence(&p, &p)).abs() < 1e-12);
    }

    #[test]
    fn js_divergence_bounded() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let js = js_divergence(&p, &q);
        assert!(js > 0.0);
        assert!(js <= 1.0, "JS divergence ≤ 1 bit");
    }

    #[test]
    fn min_entropy_leq_shannon() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let h_min = min_entropy(&p);
        let h_shannon = shannon_entropy(&p);
        assert!(h_min <= h_shannon + 1e-12);
    }

    #[test]
    fn empty_vector_errors() {
        assert!(VerificationEntropy::from_probabilities(&[]).is_err());
    }

    #[test]
    fn negative_probability_errors() {
        assert!(VerificationEntropy::from_probabilities(&[-0.1, 0.5]).is_err());
    }
}
