//! Entropy gradient descent: suggest code changes that reduce H.
//!
//! Split large functions, add missing tests, reduce branching.

use crate::entropy::shannon_entropy;
use crate::flow::ModuleId;

/// A concrete suggestion for reducing verification entropy.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientSuggestion {
    pub module: ModuleId,
    pub action: String,
    pub expected_reduction: f64,
    pub priority: f64,
}

/// Engine for computing entropy-reducing suggestions.
#[derive(Debug, Clone)]
pub struct GradientDescent {
    /// Current entropy per module.
    pub module_entropy: std::collections::HashMap<ModuleId, f64>,
    /// Branching factor per module (e.g. cyclomatic complexity).
    pub branch_factor: std::collections::HashMap<ModuleId, f64>,
    /// Test coverage per module ∈ [0, 1].
    pub test_coverage: std::collections::HashMap<ModuleId, f64>,
}

impl GradientDescent {
    /// Create a new gradient descent engine.
    pub fn new() -> Self {
        Self {
            module_entropy: Default::default(),
            branch_factor: Default::default(),
            test_coverage: Default::default(),
        }
    }

    /// Set entropy for a module.
    pub fn with_entropy(mut self, module: &str, h: f64) -> Self {
        self.module_entropy.insert(module.into(), h);
        self
    }

    /// Set branching factor for a module.
    pub fn with_branches(mut self, module: &str, b: f64) -> Self {
        self.branch_factor.insert(module.into(), b);
        self
    }

    /// Set test coverage for a module.
    pub fn with_coverage(mut self, module: &str, c: f64) -> Self {
        self.test_coverage.insert(module.into(), c);
        self
    }

    /// Compute the entropy gradient: partial derivative of total H w.r.t. each module.
    ///
    /// Higher gradient → more entropy reduction potential.
    pub fn gradient(&self) -> Vec<(ModuleId, f64)> {
        let mut grad: Vec<(ModuleId, f64)> = self
            .module_entropy
            .iter()
            .map(|(m, &h)| {
                let branches = self.branch_factor.get(m).copied().unwrap_or(1.0);
                let coverage = self.test_coverage.get(m).copied().unwrap_or(1.0);
                // Gradient magnitude: entropy × branching / coverage
                // High entropy + high branching + low coverage → biggest opportunity
                let g = h * branches / coverage.max(0.01);
                (m.clone(), g)
            })
            .collect();
        grad.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        grad
    }

    /// Generate actionable suggestions sorted by expected impact.
    pub fn suggest(&self) -> Vec<GradientSuggestion> {
        let grad = self.gradient();
        let mut suggestions = Vec::new();

        for (module, g) in &grad {
            let coverage = self.test_coverage.get(module).copied().unwrap_or(1.0);
            let branches = self.branch_factor.get(module).copied().unwrap_or(1.0);
            let h = self.module_entropy.get(module).copied().unwrap_or(0.0);

            // Low coverage → suggest adding tests
            if coverage < 0.8 {
                let uncov = 1.0 - coverage;
                suggestions.push(GradientSuggestion {
                    module: module.clone(),
                    action: format!(
                        "Add tests to '{}': coverage is {:.0}%, expected ~{:.1} new paths to cover",
                        module, coverage * 100.0, branches * uncov
                    ),
                    expected_reduction: h * uncov * 0.5,
                    priority: g * uncov,
                });
            }

            // High branching → suggest splitting
            if branches > 5.0 {
                let excess = branches - 5.0;
                suggestions.push(GradientSuggestion {
                    module: module.clone(),
                    action: format!(
                        "Split '{}': branching factor {:.0} exceeds threshold, refactor into {} smaller functions",
                        module, branches, (branches / 3.0).ceil() as usize
                    ),
                    expected_reduction: h * (excess / branches) * 0.3,
                    priority: g * (excess / branches),
                });
            }

            // High absolute entropy → general reduction
            if h > 2.0 {
                suggestions.push(GradientSuggestion {
                    module: module.clone(),
                    action: format!(
                        "Reduce entropy in '{}': H={:.2} bits is high, consider simplifying control flow",
                        module, h
                    ),
                    expected_reduction: (h - 2.0) * 0.2,
                    priority: g * 0.5,
                });
            }
        }

        suggestions.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
        suggestions
    }
}

/// Simulate one step of entropy gradient descent.
/// Returns updated probability distribution after a step of size `lr`.
pub fn descend(probs: &[f64], lr: f64) -> Vec<f64> {
    let h = shannon_entropy(probs);
    if h < 1e-12 {
        return probs.to_vec();
    }
    // Gradient of H w.r.t. pᵢ: ∂H/∂pᵢ = -(log₂ pᵢ + 1)
    let grad: Vec<f64> = probs
        .iter()
        .map(|&p| {
            if p > 0.0 {
                -(p.log2() + 1.0)
            } else {
                0.0
            }
        })
        .collect();

    // Step: p_new = p - lr * grad, then project back to simplex
    let mut new_p: Vec<f64> = probs.iter().zip(grad.iter()).map(|(&p, &g)| p - lr * g).collect();

    // Project onto probability simplex: clip negatives, renormalise
    for p in &mut new_p {
        *p = p.max(0.0);
    }
    let total: f64 = new_p.iter().sum();
    if total > 0.0 {
        for p in &mut new_p {
            *p /= total;
        }
    }
    new_p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_highlights_highest_entropy() {
        let gd = GradientDescent::new()
            .with_entropy("big_mod", 3.5)
            .with_entropy("small_mod", 0.5)
            .with_branches("big_mod", 10.0)
            .with_branches("small_mod", 2.0)
            .with_coverage("big_mod", 0.3)
            .with_coverage("small_mod", 0.9);

        let grad = gd.gradient();
        assert_eq!(grad[0].0, "big_mod");
        assert!(grad[0].1 > grad[1].1);
    }

    #[test]
    fn suggest_adds_tests_for_low_coverage() {
        let gd = GradientDescent::new()
            .with_entropy("mod_a", 2.0)
            .with_coverage("mod_a", 0.4)
            .with_branches("mod_a", 3.0);

        let suggestions = gd.suggest();
        assert!(suggestions.iter().any(|s| s.action.contains("Add tests")));
    }

    #[test]
    fn suggest_splits_high_branches() {
        let gd = GradientDescent::new()
            .with_entropy("mod_b", 2.5)
            .with_coverage("mod_b", 0.9)
            .with_branches("mod_b", 12.0);

        let suggestions = gd.suggest();
        assert!(suggestions.iter().any(|s| s.action.contains("Split")));
    }

    #[test]
    fn descent_reduces_entropy() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let h_before = shannon_entropy(&p);
        let p_new = descend(&p, 0.01);
        let h_after = shannon_entropy(&p_new);
        assert!(h_after < h_before + 1e-12, "descent should not increase entropy");
    }

    #[test]
    fn descent_preserves_simplex() {
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let p_new = descend(&p, 0.05);
        let sum: f64 = p_new.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "probabilities should sum to 1");
        for &pi in &p_new {
            assert!(pi >= 0.0, "probabilities must be non-negative");
        }
    }
}
