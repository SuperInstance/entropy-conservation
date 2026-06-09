//! Tutorial: Entropy conservation in agent systems
//!
//! Shows VerificationEntropy, conservation checking, entropy flow networks.

use entropy_conservation::entropy::{VerificationEntropy, shannon_entropy, kl_divergence};
use entropy_conservation::conservation::ConservationReport;
use entropy_conservation::flow::{EntropyFlow, FlowNetwork};

fn main() {
    println!("=== Entropy Conservation Tutorial ===\n");

    // Part 1: Shannon entropy of agent distributions
    println!("Part 1: Measuring agent decision entropy");
    let fair = vec![0.25, 0.25, 0.25, 0.25];
    let biased = vec![0.7, 0.1, 0.1, 0.1];
    
    println!("  Fair agent:  H = {:.3} bits (max: {:.3})", shannon_entropy(&fair), VerificationEntropy::max_shannon(4));
    println!("  Biased agent: H = {:.3} bits", shannon_entropy(&biased));
    println!();

    // Part 2: VerificationEntropy from probabilities
    println!("Part 2: VerificationEntropy");
    let ve = VerificationEntropy::from_probabilities(&fair).unwrap();
    println!("  Normalised: {:.3}", ve.normalised());
    let ve_counts = VerificationEntropy::from_counts(&[100, 50, 30, 20]).unwrap();
    println!("  From counts [100,50,30,20]: normalised = {:.3}", ve_counts.normalised());
    println!();

    // Part 3: KL divergence between policies
    println!("Part 3: KL divergence");
    println!("  KL(biased || fair) = {:.4}", kl_divergence(&biased, &fair));
    println!();

    // Part 4: Conservation check via snapshots
    println!("Part 4: Conservation report");
    let ve_before = VerificationEntropy::from_probabilities(&fair).unwrap();
    let ve_after = VerificationEntropy::from_probabilities(&vec![0.24, 0.26, 0.25, 0.25]).unwrap();
    let report = ConservationReport::from_snapshots(&[ve_before, ve_after]);
    println!("  Conserved: {}", report.is_conserved());
    println!("  Total violation: {:.6}", report.total_violation());
    println!();

    // Part 5: Entropy flow network
    println!("Part 5: Entropy flow network");
    let flows = vec![
        EntropyFlow { source: "parser".into(), sink: "optimizer".into(), rate: 0.3 },
        EntropyFlow { source: "optimizer".into(), sink: "codegen".into(), rate: 0.25 },
        EntropyFlow { source: "codegen".into(), sink: "parser".into(), rate: 0.25 },
    ];
    let network = FlowNetwork::new(flows);
    let (conserved, leakage) = network.check_conservation();
    println!("  Network conserved: {}", conserved);
    println!("  Leakage: {:.4}", leakage);
}
