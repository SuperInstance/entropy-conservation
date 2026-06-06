//! Spectral analysis of the entropy graph.
//!
//! Eigenvalues of the entropy Laplacian, heat kernel,
//! and spectral gap bounds on mixing time.

use crate::flow::FlowNetwork;

/// Result of spectral analysis on the verification graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpectralAnalysis {
    /// Eigenvalues of the normalised Laplacian, sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Spectral gap: λ₁ (the smallest non-zero eigenvalue).
    pub spectral_gap: f64,
    /// Algebraic connectivity (= λ₁ of the Laplacian).
    pub algebraic_connectivity: f64,
    /// Upper bound on mixing time from the spectral gap.
    pub mixing_time_bound: f64,
    /// Number of connected components (count of zero eigenvalues).
    pub components: usize,
}

impl SpectralAnalysis {
    /// Perform spectral analysis on a flow network's Laplacian.
    pub fn from_network(network: &FlowNetwork) -> Self {
        let n = network.modules.len();
        if n == 0 {
            return Self {
                eigenvalues: vec![],
                spectral_gap: 0.0,
                algebraic_connectivity: 0.0,
                mixing_time_bound: f64::INFINITY,
                components: 0,
            };
        }

        let laplacian = network.laplacian();
        let eigenvalues = power_iteration_eigenvalues(&laplacian, n.min(20));

        // Count near-zero eigenvalues → connected components
        let components = eigenvalues.iter().filter(|&&e| e.abs() < 1e-8).count().max(1);

        // Spectral gap = smallest non-zero eigenvalue
        let spectral_gap = eigenvalues
            .iter()
            .filter(|&&e| e > 1e-8)
            .cloned()
            .next()
            .unwrap_or(0.0);

        // Mixing time bound: τ ≤ (1/λ₁) log(1/ε)  with ε=0.01
        let mixing_time_bound = if spectral_gap > 1e-12 {
            (1.0 / spectral_gap) * (100.0_f64).ln()
        } else {
            f64::INFINITY
        };

        Self {
            eigenvalues,
            spectral_gap,
            algebraic_connectivity: spectral_gap,
            mixing_time_bound,
            components,
        }
    }

    /// Perform spectral analysis on an arbitrary adjacency matrix.
    pub fn from_adjacency(adj: &ndarray::Array2<f64>) -> Self {
        let n = adj.nrows();
        if n == 0 {
            return Self {
                eigenvalues: vec![],
                spectral_gap: 0.0,
                algebraic_connectivity: 0.0,
                mixing_time_bound: f64::INFINITY,
                components: 0,
            };
        }

        // Build Laplacian
        let mut laplacian = ndarray::Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += adj[[i, j]];
            }
            laplacian[[i, i]] = row_sum;
            for j in 0..n {
                laplacian[[i, j]] -= adj[[i, j]];
            }
        }

        let eigenvalues = power_iteration_eigenvalues(&laplacian, n.min(20));
        let components = eigenvalues.iter().filter(|&&e| e.abs() < 1e-8).count().max(1);
        let spectral_gap = eigenvalues
            .iter()
            .filter(|&&e| e > 1e-8)
            .cloned()
            .next()
            .unwrap_or(0.0);
        let mixing_time_bound = if spectral_gap > 1e-12 {
            (1.0 / spectral_gap) * (100.0_f64).ln()
        } else {
            f64::INFINITY
        };

        Self {
            eigenvalues,
            spectral_gap,
            algebraic_connectivity: spectral_gap,
            mixing_time_bound,
            components,
        }
    }

    /// Heat kernel: H(t) = exp(-tL). Approximated via eigendecomposition.
    ///
    /// Returns the heat kernel trace at time t.
    pub fn heat_kernel_trace(&self, t: f64) -> f64 {
        self.eigenvalues.iter().map(|&lambda| (-t * lambda).exp()).sum()
    }
}

/// Compute eigenvalues using QR iteration (simplified).
/// Returns eigenvalues sorted in ascending order.
fn power_iteration_eigenvalues(mat: &ndarray::Array2<f64>, max_eigs: usize) -> Vec<f64> {
    let n = mat.nrows();
    if n == 0 {
        return vec![];
    }

    // For small matrices, use direct QR iteration
    let mut a = mat.clone();
    let iterations = 50;

    for _ in 0..iterations {
        // QR decomposition via Gram-Schmidt
        let (q, r) = qr_decompose(&a);
        a = r.dot(&q);
    }

    // Extract diagonal as approximate eigenvalues
    let mut eigs: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();

    // Handle 2x2 blocks (complex eigenvalues → take real part)
    // For simplicity, just sort and return
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate if needed
    eigs.truncate(max_eigs);
    eigs
}

/// Simple QR decomposition using modified Gram-Schmidt.
fn qr_decompose(a: &ndarray::Array2<f64>) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    let n = a.nrows();
    let m = a.ncols();
    let mut q = ndarray::Array2::<f64>::zeros((n, m));
    let mut r = ndarray::Array2::<f64>::zeros((m, m));

    for j in 0..m {
        // v = a[:, j]
        let mut v: Vec<f64> = (0..n).map(|i| a[[i, j]]).collect();

        for i in 0..j {
            // r[i,j] = q[:,i] · v
            let dot: f64 = (0..n).map(|k| q[[k, i]] * v[k]).sum();
            r[[i, j]] = dot;
            for k in 0..n {
                v[k] -= dot * q[[k, i]];
            }
        }

        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r[[j, j]] = norm;

        if norm > 1e-15 {
            for k in 0..n {
                q[[k, j]] = v[k] / norm;
            }
        }
    }

    (q, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::EntropyFlow;

    fn cycle_network() -> FlowNetwork {
        FlowNetwork::new(vec![
            EntropyFlow { source: "A".into(), sink: "B".into(), rate: 1.0 },
            EntropyFlow { source: "B".into(), sink: "C".into(), rate: 1.0 },
            EntropyFlow { source: "C".into(), sink: "D".into(), rate: 1.0 },
            EntropyFlow { source: "D".into(), sink: "A".into(), rate: 1.0 },
        ])
    }

    #[test]
    fn spectral_gap_positive_for_connected() {
        let net = cycle_network();
        let analysis = SpectralAnalysis::from_network(&net);
        assert!(
            analysis.spectral_gap > 0.01,
            "connected graph should have positive spectral gap: {}",
            analysis.spectral_gap
        );
    }

    #[test]
    fn one_component_for_connected() {
        let net = cycle_network();
        let analysis = SpectralAnalysis::from_network(&net);
        assert_eq!(analysis.components, 1);
    }

    #[test]
    fn mixing_time_finite_for_connected() {
        let net = cycle_network();
        let analysis = SpectralAnalysis::from_network(&net);
        assert!(analysis.mixing_time_bound.is_finite());
    }

    #[test]
    fn eigenvalues_non_negative() {
        let net = cycle_network();
        let analysis = SpectralAnalysis::from_network(&net);
        for &e in &analysis.eigenvalues {
            assert!(e >= -1e-6, "Laplacian eigenvalues must be ≥ 0, got {e}");
        }
    }

    #[test]
    fn smallest_eigenvalue_is_zero() {
        let net = cycle_network();
        let analysis = SpectralAnalysis::from_network(&net);
        assert!(
            analysis.eigenvalues[0].abs() < 0.1,
            "smallest Laplacian eigenvalue should be ≈ 0"
        );
    }

    #[test]
    fn heat_kernel_trace_decreases() {
        let net = cycle_network();
        let analysis = SpectralAnalysis::from_network(&net);
        let trace_t1 = analysis.heat_kernel_trace(0.1);
        let trace_t10 = analysis.heat_kernel_trace(1.0);
        assert!(
            trace_t10 <= trace_t1 + 1e-10,
            "heat kernel trace should decrease with time"
        );
    }

    #[test]
    fn spectral_analysis_from_adjacency() {
        let mut adj = ndarray::Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        let analysis = SpectralAnalysis::from_adjacency(&adj);
        assert_eq!(analysis.components, 1);
        assert!(analysis.eigenvalues[0].abs() < 0.5);
    }
}
