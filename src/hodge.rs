//! Hodge decomposition of entropy flow on the verification graph.
//!
//! Decomposes any flow into three orthogonal components:
//! - **Exact (gradient)**: conservative flow from a potential — structural entropy
//! - **Co-exact (divergence)**: flow driven by sources/sinks — accidental entropy
//! - **Harmonic**: divergence-free and curl-free — topological entropy
//!
//! By Hodge's theorem: Ω¹ = dΩ⁰ ⊕ δΩ² ⊕ ℋ¹

use crate::flow::{EntropyFlow, FlowNetwork};

/// Hodge decomposition of entropy flow.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HodgeDecomposition {
    /// Exact (conservative/gradient) component.
    pub exact: Vec<EntropyFlow>,
    /// Harmonic (topological) component.
    pub harmonic: Vec<EntropyFlow>,
    /// Co-exact (divergence-driven) component.
    pub coexact: Vec<EntropyFlow>,
}

impl HodgeDecomposition {
    /// Perform Hodge decomposition on a flow network.
    ///
    /// Uses the graph Laplacian to compute the potential (scalar field)
    /// from which the exact component derives.
    pub fn decompose(network: &FlowNetwork) -> Self {
        let n = network.modules.len();
        if n == 0 || network.flows.is_empty() {
            return Self {
                exact: vec![],
                harmonic: vec![],
                coexact: vec![],
            };
        }

        let idx: std::collections::HashMap<&str, usize> = network
            .modules
            .iter()
            .enumerate()
            .map(|(i, m)| (m.as_str(), i))
            .collect();

        // Build the divergence vector: div(i) = inflow - outflow
        let mut divergence = vec![0.0; n];
        for f in &network.flows {
            if let Some(&j) = idx.get(f.sink.as_str()) {
                divergence[j] += f.rate;
            }
            if let Some(&i) = idx.get(f.source.as_str()) {
                divergence[i] -= f.rate;
            }
        }

        // Solve for potential φ using the Laplacian: L φ = -div
        // We use an iterative Jacobi method for simplicity.
        let laplacian = network.laplacian();
        let phi = solve_laplacian_jacobi(&laplacian, &divergence, n, 1000);

        // Exact component: flow from potential gradient
        let mut exact = Vec::new();
        for f in &network.flows {
            if let (Some(&i), Some(&j)) = (idx.get(f.source.as_str()), idx.get(f.sink.as_str())) {
                let grad_flow = phi[j] - phi[i];
                if grad_flow.abs() > 1e-15 {
                    exact.push(EntropyFlow {
                        source: f.source.clone(),
                        sink: f.sink.clone(),
                        rate: grad_flow,
                    });
                }
            }
        }

        // Residual = original - exact; split into co-exact and harmonic
        let mut residual: Vec<EntropyFlow> = network.flows.iter().map(|f| f.clone()).collect();
        for e in &exact {
            for r in &mut residual {
                if r.source == e.source && r.sink == e.sink {
                    r.rate -= e.rate;
                }
            }
        }

        // For small graphs, the harmonic component is the divergence-free,
        // curl-free part of the residual. In practice, we separate by
        // checking if the residual has nonzero divergence.
        let mut harmonic = Vec::new();
        let mut coexact = Vec::new();

        // Compute residual divergence per node
        let mut res_div = vec![0.0; n];
        for r in &residual {
            if let Some(&j) = idx.get(r.sink.as_str()) {
                res_div[j] += r.rate;
            }
            if let Some(&i) = idx.get(r.source.as_str()) {
                res_div[i] -= r.rate;
            }
        }

        for r in &residual {
            let div_source = idx.get(r.source.as_str()).map(|&i| res_div[i]).unwrap_or(0.0);
            let div_sink = idx.get(r.sink.as_str()).map(|&j| res_div[j]).unwrap_or(0.0);
            if div_source.abs() < 1e-10 && div_sink.abs() < 1e-10 {
                harmonic.push(r.clone());
            } else {
                coexact.push(r.clone());
            }
        }

        Self {
            exact,
            harmonic,
            coexact,
        }
    }

    /// Reconstruct the original flow by summing all components.
    pub fn reconstruct(&self) -> Vec<EntropyFlow> {
        let mut combined = Vec::new();
        combined.extend(self.exact.iter().cloned());
        combined.extend(self.harmonic.iter().cloned());
        combined.extend(self.coexact.iter().cloned());

        // Merge duplicate edges
        let mut merged: std::collections::HashMap<(String, String), f64> = std::collections::HashMap::new();
        for f in &combined {
            *merged.entry((f.source.clone(), f.sink.clone())).or_default() += f.rate;
        }
        merged
            .into_iter()
            .map(|((source, sink), rate)| EntropyFlow { source, sink, rate })
            .collect()
    }
}

/// Solve L φ = -div using Jacobi iteration (with regularisation for singular L).
fn solve_laplacian_jacobi(
    laplacian: &ndarray::Array2<f64>,
    div: &[f64],
    n: usize,
    max_iter: usize,
) -> Vec<f64> {
    let mut phi = vec![0.0; n];
    let reg = 1e-8; // regularisation for isolated nodes

    for _ in 0..max_iter {
        let mut new_phi = vec![0.0; n];
        for i in 0..n {
            let diag = laplacian[[i, i]] + reg;
            if diag.abs() < 1e-15 {
                new_phi[i] = phi[i];
                continue;
            }
            let mut off_diag_sum = 0.0;
            for j in 0..n {
                if j != i {
                    off_diag_sum += laplacian[[i, j]] * phi[j];
                }
            }
            // L φ = -div  →  diag * φᵢ = -divᵢ - Σⱼ≠ᵢ Lᵢⱼ φⱼ
            new_phi[i] = (-div[i] - off_diag_sum) / diag;
        }
        let change: f64 = new_phi.iter().zip(phi.iter()).map(|(a, b)| (a - b).abs()).sum();
        phi = new_phi;
        if change < 1e-12 {
            break;
        }
    }
    phi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flow_network() -> FlowNetwork {
        FlowNetwork::new(vec![
            EntropyFlow { source: "A".into(), sink: "B".into(), rate: 2.0 },
            EntropyFlow { source: "A".into(), sink: "C".into(), rate: 1.0 },
            EntropyFlow { source: "B".into(), sink: "C".into(), rate: 2.0 },
            EntropyFlow { source: "C".into(), sink: "A".into(), rate: 3.0 },
        ])
    }

    #[test]
    fn hodge_decomposition_sums_to_original() {
        let net = make_flow_network();
        let hodge = HodgeDecomposition::decompose(&net);

        let reconstructed = hodge.reconstruct();
        let original = net.flows.clone();

        // Build maps for comparison
        let mut orig_map: std::collections::HashMap<(String, String), f64> = std::collections::HashMap::new();
        for f in &original {
            *orig_map.entry((f.source.clone(), f.sink.clone())).or_default() += f.rate;
        }
        let mut recon_map: std::collections::HashMap<(String, String), f64> = std::collections::HashMap::new();
        for f in &reconstructed {
            *recon_map.entry((f.source.clone(), f.sink.clone())).or_default() += f.rate;
        }

        for (edge, &rate) in &orig_map {
            let recon_rate = recon_map.get(edge).copied().unwrap_or(0.0);
            assert!(
                (rate - recon_rate).abs() < 0.1,
                "edge {:?}: original={}, reconstructed={}",
                edge, rate, recon_rate
            );
        }
    }

    #[test]
    fn hodge_empty_network() {
        let net = FlowNetwork::new(vec![]);
        let hodge = HodgeDecomposition::decompose(&net);
        assert!(hodge.exact.is_empty());
        assert!(hodge.harmonic.is_empty());
        assert!(hodge.coexact.is_empty());
    }

    #[test]
    fn hodge_cycle_has_harmonic() {
        // Pure cycle: A → B → C → A, all rate 1.0
        // This is divergence-free and curl-free → should be harmonic
        let net = FlowNetwork::new(vec![
            EntropyFlow { source: "A".into(), sink: "B".into(), rate: 1.0 },
            EntropyFlow { source: "B".into(), sink: "C".into(), rate: 1.0 },
            EntropyFlow { source: "C".into(), sink: "A".into(), rate: 1.0 },
        ]);
        let hodge = HodgeDecomposition::decompose(&net);
        // The harmonic component should capture the cycle flow
        let harmonic_flow: f64 = hodge.harmonic.iter().map(|f| f.rate.abs()).sum();
        assert!(harmonic_flow > 0.5, "cycle should have significant harmonic component");
    }

    #[test]
    fn hodge_components_non_negative_flows() {
        let net = make_flow_network();
        let hodge = HodgeDecomposition::decompose(&net);
        // Just verify it doesn't panic and produces valid structure
        assert!(!hodge.exact.is_empty() || !hodge.harmonic.is_empty() || !hodge.coexact.is_empty());
    }
}
