# entropy-conservation

[![crates.io](https://img.shields.io/crates/v/entropy-conservation.svg)](https://crates.io/crates/entropy-conservation)
[![docs.rs](https://docs.rs/entropy-conservation/badge.svg)](https://docs.rs/entropy-conservation)

**Conservation of Verification Entropy** — the mathematical framework behind the meta-law discovered across all PLATO/SuperInstance experiments.

## The Meta-Law

> Every closed system of verification (tests, proofs, type checks) has a conserved quantity **H** — the *verification entropy*. The conservation law states:
>
> **dH/dt ≤ 0**
>
> Verification entropy never increases in a well-structured system.

### What is Verification Entropy?

Given a system with *n* verification paths (test branches, proof obligations, type-check branches), each exercised with probability *pᵢ*, the verification entropy is:

```
H = -Σ pᵢ log₂ pᵢ
```

This is Shannon entropy applied to the probability distribution over verification paths. It measures the *uncertainty* inherent in the system's verification structure.

**Conservation means:** As code evolves, the total verification entropy of a well-maintained system decreases or stays constant. Entropy-increasing changes — added untested branches, increased cyclomatic complexity, dead verification paths — are *violations* of the conservation law.

## Mathematical Framework

This library implements seven interconnected modules:

### 1. Entropy (`entropy`)

Computes Shannon, Rényi (order α), and Tsallis entropy from test coverage vectors.

```rust
use entropy_conservation::entropy::{VerificationEntropy, shannon_entropy, renyi_entropy};

// From a coverage vector
let ve = VerificationEntropy::from_counts(&[10, 20, 30, 40])?;
println!("Shannon entropy: {:.4} bits", ve.shannon);

// Rényi entropy converges to Shannon as α→1
let p = vec![0.1, 0.2, 0.3, 0.4];
let shannon = shannon_entropy(&p);
let renyi_1 = renyi_entropy(&p, 1.0000001)?;
assert!((shannon - renyi_1).abs() < 1e-4);
```

### 2. Conservation (`conservation`)

Tracks H over code changes, detects entropy-increasing commits, computes the entropy gradient.

```rust
use entropy_conservation::conservation::check_conservation;

let before = vec![0.25, 0.25, 0.25, 0.25]; // uniform → H = 2.0 bits
let after  = vec![0.5, 0.3, 0.1, 0.1];     // skewed   → H < 2.0 bits
let report = check_conservation(&before, &after)?;
assert!(report.is_conserved()); // entropy decreased ✓
```

### 3. Flow (`flow`)

Models entropy flow between modules as a directed graph. Enforces **Kirchhoff's current law**: at every node, inflow = outflow.

```rust
use entropy_conservation::flow::{FlowNetwork, EntropyFlow};

let net = FlowNetwork::new(vec![
    EntropyFlow { source: "parser".into(), sink: "ast".into(),   rate: 2.0 },
    EntropyFlow { source: "ast".into(),    sink: "typeck".into(), rate: 2.0 },
    EntropyFlow { source: "typeck".into(), sink: "parser".into(), rate: 2.0 },
]);
let (conserved, _) = net.check_conservation();
assert!(conserved); // Kirchhoff's law holds
```

### 4. Gradient (`gradient`)

Entropy gradient descent: suggests code changes that reduce H — split large functions, add missing tests, reduce branching.

```rust
use entropy_conservation::gradient::GradientDescent;

let gd = GradientDescent::new()
    .with_entropy("big_mod", 3.5)
    .with_branches("big_mod", 12.0)
    .with_coverage("big_mod", 0.3);

for suggestion in gd.suggest() {
    println!("{}: {} (−{:.2} bits)", suggestion.module, suggestion.action, suggestion.expected_reduction);
}
```

### 5. Hodge Decomposition (`hodge`)

By **Hodge's theorem**, any flow on a graph decomposes orthogonally:

```
Ω¹ = dΩ⁰ ⊕ δΩ² ⊕ ℋ¹
```

- **Exact (gradient)**: conservative flow from a potential — *structural* entropy
- **Co-exact (divergence)**: flow driven by sources/sinks — *accidental* entropy
- **Harmonic**: divergence-free and curl-free — *topological* entropy

```rust
use entropy_conservation::hodge::HodgeDecomposition;

let hodge = HodgeDecomposition::decompose(&network);
println!("Structural entropy: {} exact flows", hodge.exact.len());
println!("Accidental entropy: {} co-exact flows", hodge.coexact.len());
println!("Topological entropy: {} harmonic flows", hodge.harmonic.len());
```

### 6. Persistence Homology (`persistence`)

Computes Betti numbers and persistence diagrams for the verification space, revealing topological structure across scales.

```rust
use entropy_conservation::persistence::{PersistenceDiagram, entropy_landscape};

let landscape = entropy_landscape(&[
    vec![0.5, 0.5],   // module A
    vec![0.3, 0.7],   // module B
    vec![0.1, 0.9],   // module C
]);
let pd = PersistenceDiagram::from_ndarray_matrix(&landscape);
println!("β₀ = {} (connected components)", pd.betti_numbers[0]);
println!("β₁ = {} (cycles)", pd.betti_numbers[1]);
assert!(pd.is_valid()); // all points above diagonal
```

### 7. Spectral Analysis (`spectrum`)

Eigenvalues of the entropy Laplacian, heat kernel on the verification graph, and spectral-gap bounds on mixing time.

```rust
use entropy_conservation::spectrum::SpectralAnalysis;

let analysis = SpectralAnalysis::from_network(&network);
println!("Spectral gap λ₁ = {:.4}", analysis.spectral_gap);
println!("Mixing time ≤ {:.2}", analysis.mixing_time_bound);
println!("Heat kernel trace at t=0.1: {:.4}", analysis.heat_kernel_trace(0.1));
```

## Key Properties (tested)

| Property | Statement |
|----------|-----------|
| Non-negativity | H ≥ 0 for any probability distribution |
| Maximum entropy | H ≤ log₂ n (achieved by uniform distribution) |
| Conservation | dH/dt ≤ 0 in well-structured evolution |
| Kirchhoff's law | Total inflow = total outflow at every node |
| Hodge decomposition | exact + harmonic + co-exact = original flow |
| Persistence validity | All (birth, death) pairs satisfy death ≥ birth |
| Rényi convergence | H_α → H_shannon as α → 1 |
| Rényi monotonicity | H_α is non-increasing in α |
| Min-entropy bound | H_∞ ≤ H_shannon |
| Spectral gap | λ₁ > 0 ⟺ graph is connected |
| Laplacian rows sum to 0 | Σⱼ Lᵢⱼ = 0 for all i |

## Core Types

```rust
struct VerificationEntropy {
    shannon: f64,
    renyi: BTreeMap<OrderedFloat, f64>,
    tsallis: f64,
}

struct EntropyFlow {
    source: ModuleId,
    sink: ModuleId,
    rate: f64,
}

struct ConservationReport {
    before: f64,
    after: f64,
    delta: f64,
    violations: Vec<EntropyViolation>,
}

struct HodgeDecomposition {
    exact: Vec<EntropyFlow>,
    harmonic: Vec<EntropyFlow>,
    coexact: Vec<EntropyFlow>,
}

struct PersistenceDiagram {
    points: Vec<(f64, f64)>,
    betti_numbers: Vec<usize>,
}
```

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
entropy-conservation = "0.1"
```

## Theoretical Background

### Why This Matters

The Conservation of Verification Entropy emerged as a pattern across hundreds of PLATO/SuperInstance experiments. Systems that maintained or decreased their verification entropy over time were consistently more reliable, more maintainable, and easier to reason about. Systems where entropy increased — through untested branches, growing cyclomatic complexity, or divergent test coverage — exhibited corresponding degradation in correctness guarantees.

### The Conservation Law as a Lagrangian

Just as Noether's theorem connects conservation laws to symmetries in physics, the conservation of verification entropy connects to a symmetry of the verification system: *invariance under reordering of verification paths*. When the system has this symmetry, dH/dt = 0 exactly. When the symmetry is broken (paths are added unevenly, coverage becomes non-uniform), dH/dt < 0 — the system naturally tends toward lower entropy as the "path of least resistance" dominates.

### Hodge Theory and Structural vs. Accidental Entropy

The Hodge decomposition distinguishes between:
- **Structural entropy** (exact component): inherent in the system's architecture. Cannot be removed without restructuring.
- **Accidental entropy** (co-exact component): introduced by implementation choices. Can be reduced by refactoring.
- **Topological entropy** (harmonic component): arising from the graph topology (cycles in dependencies). Invariant under continuous deformation.

### Persistence Homology and Scale

Persistence diagrams reveal *at what scale* topological features of the verification space appear and disappear. Long bars in the diagram represent robust structural features; short bars represent noise.

## License

MIT
