//! Entropy flow between modules: which modules export/import entropy,
//! flow conservation in dependency graphs (Kirchhoff's current law).

use std::collections::HashMap;


/// Identifier for a module in the dependency graph.
pub type ModuleId = String;

/// A directed flow of entropy from one module to another.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntropyFlow {
    pub source: ModuleId,
    pub sink: ModuleId,
    pub rate: f64,
}

/// A flow network on a module dependency graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlowNetwork {
    /// All module IDs.
    pub modules: Vec<ModuleId>,
    /// Directed entropy flows.
    pub flows: Vec<EntropyFlow>,
}

impl FlowNetwork {
    /// Build a flow network from a list of flows.
    pub fn new(flows: Vec<EntropyFlow>) -> Self {
        let mut module_set = std::collections::HashSet::new();
        for f in &flows {
            module_set.insert(f.source.clone());
            module_set.insert(f.sink.clone());
        }
        let mut modules: Vec<_> = module_set.into_iter().collect();
        modules.sort();
        Self { modules, flows }
    }

    /// Compute net flow at each node: outflow - inflow.
    ///
    /// For a conserved flow, net flow = 0 at every internal node.
    pub fn net_flow(&self) -> HashMap<ModuleId, f64> {
        let mut net: HashMap<ModuleId, f64> = HashMap::new();
        for m in &self.modules {
            net.insert(m.clone(), 0.0);
        }
        for f in &self.flows {
            *net.get_mut(&f.source).unwrap() -= f.rate;
            *net.get_mut(&f.sink).unwrap() += f.rate;
        }
        net
    }

    /// Check Kirchhoff's current law: total inflow = total outflow at each node.
    ///
    /// Returns (is_conserved, max_violation).
    pub fn check_conservation(&self) -> (bool, f64) {
        let net = self.net_flow();
        let max_violation = net.values().map(|v| v.abs()).fold(0.0_f64, f64::max);
        (max_violation < 1e-10, max_violation)
    }

    /// Total flow magnitude.
    pub fn total_flow(&self) -> f64 {
        self.flows.iter().map(|f| f.rate).sum()
    }

    /// Build the flow adjacency matrix (n×n) for spectral analysis.
    pub fn adjacency_matrix(&self) -> (usize, ndarray::Array2<f64>) {
        let n = self.modules.len();
        let mut mat = ndarray::Array2::<f64>::zeros((n, n));
        let idx: HashMap<&ModuleId, usize> = self.modules.iter().enumerate().map(|(i, m)| (m, i)).collect();

        for f in &self.flows {
            if let (Some(&i), Some(&j)) = (idx.get(&f.source), idx.get(&f.sink)) {
                mat[[i, j]] += f.rate;
            }
        }
        (n, mat)
    }

    /// Compute the graph Laplacian L = D - A.
    pub fn laplacian(&self) -> ndarray::Array2<f64> {
        let (n, a) = self.adjacency_matrix();
        let mut d = ndarray::Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += a[[i, j]];
            }
            d[[i, i]] = row_sum;
        }
        d - a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kirchhoff_conserved_cycle() {
        // A → B → C → A, equal flow
        let net = FlowNetwork::new(vec![
            EntropyFlow { source: "A".into(), sink: "B".into(), rate: 1.0 },
            EntropyFlow { source: "B".into(), sink: "C".into(), rate: 1.0 },
            EntropyFlow { source: "C".into(), sink: "A".into(), rate: 1.0 },
        ]);
        let (ok, violation) = net.check_conservation();
        assert!(ok, "cycle should be conserved, max violation = {violation}");
    }

    #[test]
    fn kirchhoff_source_sink_violates() {
        // A → B (no return)
        let net = FlowNetwork::new(vec![
            EntropyFlow { source: "A".into(), sink: "B".into(), rate: 1.0 },
        ]);
        let (ok, _) = net.check_conservation();
        assert!(!ok, "unbalanced flow should violate conservation");
    }

    #[test]
    fn kirchhoff_multi_flow_conserved() {
        // A → B (2.0), A → C (1.0), B → C (2.0), C → A (3.0)
        let net = FlowNetwork::new(vec![
            EntropyFlow { source: "A".into(), sink: "B".into(), rate: 2.0 },
            EntropyFlow { source: "A".into(), sink: "C".into(), rate: 1.0 },
            EntropyFlow { source: "B".into(), sink: "C".into(), rate: 2.0 },
            EntropyFlow { source: "C".into(), sink: "A".into(), rate: 3.0 },
        ]);
        let (ok, _) = net.check_conservation();
        assert!(ok, "balanced multi-flow should be conserved");

        let net_flow = net.net_flow();
        for (m, v) in &net_flow {
            assert!(v.abs() < 1e-10, "node {m}: net flow = {v}");
        }
    }

    #[test]
    fn laplacian_rows_sum_zero() {
        let net = FlowNetwork::new(vec![
            EntropyFlow { source: "A".into(), sink: "B".into(), rate: 1.0 },
            EntropyFlow { source: "B".into(), sink: "C".into(), rate: 1.0 },
            EntropyFlow { source: "C".into(), sink: "A".into(), rate: 1.0 },
        ]);
        let l = net.laplacian();
        let n = l.nrows();
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| l[[i, j]]).sum();
            assert!((row_sum).abs() < 1e-12, "Laplacian row {i} sums to {row_sum}, expected 0");
        }
    }

    #[test]
    fn adjacency_matrix_symmetry() {
        let net = FlowNetwork::new(vec![
            EntropyFlow { source: "A".into(), sink: "B".into(), rate: 1.0 },
            EntropyFlow { source: "B".into(), sink: "A".into(), rate: 1.0 },
        ]);
        let (_, a) = net.adjacency_matrix();
        assert!((a[[0, 1]] - a[[1, 0]]).abs() < 1e-12);
    }
}
