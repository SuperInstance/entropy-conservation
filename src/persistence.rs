//! Persistence homology of entropy landscapes.
//!
//! Computes Betti numbers and persistence diagrams for the
//! verification space, revealing the topological structure of
//! entropy across scales.

/// A persistence diagram: collection of (birth, death) pairs
/// and the resulting Betti numbers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistenceDiagram {
    /// (birth, death) pairs for each feature.
    pub points: Vec<(f64, f64)>,
    /// Betti numbers β₀, β₁, β₂, … (connected components, cycles, voids, …).
    pub betti_numbers: Vec<usize>,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
        true
    }
}

impl PersistenceDiagram {
    /// Build a persistence diagram from a distance/similarity matrix.
    ///
    /// Uses a Vietoris-Rips filtration approach.
    /// `dist` is a flattened n×n distance matrix (row-major).
    /// `n` is the number of points.
    pub fn from_distance_matrix(dist: &[f64], n: usize) -> Self {
        if n <= 1 {
            return Self {
                points: vec![],
                betti_numbers: vec![1],
            };
        }

        let mut uf = UnionFind::new(n);

        // Build edge list sorted by distance
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist.get(i * n + j).copied().unwrap_or(f64::INFINITY);
                edges.push((d, i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // H₀ persistence: track when components merge
        let mut points = Vec::new();

        for (d, i, j) in &edges {
            if d.is_infinite() {
                continue; // disconnected components never merge
            }
            if uf.find(*i) != uf.find(*j) {
                // Component merges at distance d → death of one component
                points.push((0.0, *d));
                uf.union(*i, *j);
            }
        }

        // β₀ = number of connected components
        let mut roots = std::collections::HashSet::new();
        for i in 0..n {
            roots.insert(uf.find(i));
        }
        let beta0 = roots.len();

        // Approximate β₁ = edges - vertices + components (for 1-complex)
        let beta1 = if edges.len() + beta0 > n {
            edges.len() - n + beta0
        } else {
            0
        };

        let betti_numbers = vec![beta0, beta1];

        Self { points, betti_numbers }
    }

    /// Build from an n×n ndarray distance matrix.
    pub fn from_ndarray_matrix(mat: &ndarray::Array2<f64>) -> Self {
        let n = mat.nrows();
        let dist: Vec<f64> = mat.iter().copied().collect();
        Self::from_distance_matrix(&dist, n)
    }

    /// All persistence pairs where the feature survived (death > birth).
    pub fn persistent_features(&self) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .filter(|(b, d)| d > b)
            .copied()
            .collect()
    }

    /// Total persistence: Σ |d - b|^p for each feature.
    pub fn total_persistence(&self, p: f64) -> f64 {
        self.points
            .iter()
            .filter(|(b, d)| d > b)
            .map(|(b, d)| (d - b).powf(p))
            .sum()
    }

    /// Check all points are above or on the diagonal (death ≥ birth).
    pub fn is_valid(&self) -> bool {
        self.points.iter().all(|(b, d)| *d >= *b - 1e-12)
    }
}

/// Compute the entropy landscape: pairwise Jensen-Shannon distances
/// between verification distributions for each module.
pub fn entropy_landscape(distributions: &[Vec<f64>]) -> ndarray::Array2<f64> {
    let n = distributions.len();
    let mut dist_mat = ndarray::Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let d = crate::entropy::js_divergence(&distributions[i], &distributions[j]);
            let d_sqrt = d.sqrt().max(0.0);
            dist_mat[[i, j]] = d_sqrt;
            dist_mat[[j, i]] = d_sqrt;
        }
    }
    dist_mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_diagram_points_above_diagonal() {
        let dist = vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 1.5,
            2.0, 1.5, 0.0,
        ];
        let pd = PersistenceDiagram::from_distance_matrix(&dist, 3);
        assert!(pd.is_valid(), "all points should be on or above diagonal");
    }

    #[test]
    fn betti_number_single_component() {
        let dist = vec![
            0.0, 0.1, 0.2,
            0.1, 0.0, 0.15,
            0.2, 0.15, 0.0,
        ];
        let pd = PersistenceDiagram::from_distance_matrix(&dist, 3);
        assert_eq!(pd.betti_numbers[0], 1, "connected graph → β₀ = 1");
    }

    #[test]
    fn betti_number_two_components() {
        let inf = f64::INFINITY;
        let dist = vec![
            0.0, 0.5, inf,
            0.5, 0.0, inf,
            inf, inf, 0.0,
        ];
        let pd = PersistenceDiagram::from_distance_matrix(&dist, 3);
        assert_eq!(pd.betti_numbers[0], 2, "two components → β₀ = 2");
    }

    #[test]
    fn total_persistence_non_negative() {
        let dist = vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 1.0,
            2.0, 1.0, 0.0,
        ];
        let pd = PersistenceDiagram::from_distance_matrix(&dist, 3);
        let tp = pd.total_persistence(1.0);
        assert!(tp >= 0.0);
    }

    #[test]
    fn empty_diagram() {
        let pd = PersistenceDiagram::from_distance_matrix(&[], 0);
        assert!(pd.points.is_empty());
    }

    #[test]
    fn entropy_landscape_shape() {
        let dists = vec![
            vec![0.5, 0.5],
            vec![0.3, 0.7],
            vec![0.1, 0.9],
        ];
        let mat = entropy_landscape(&dists);
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 3);
        for i in 0..3 {
            assert!((mat[[i, i]]).abs() < 1e-12);
        }
        assert!((mat[[0, 1]] - mat[[1, 0]]).abs() < 1e-12);
    }
}
