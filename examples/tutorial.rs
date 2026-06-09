//! Tutorial: Entropy conservation in agent systems
//!
//! Shows Shannon/Rényi/Tsallis entropy, KL divergence, conservation checking,
//! and entropy flow networks.

fn main() {
    println!("=== Entropy Conservation Tutorial ===\n");

    // Part 1: Shannon entropy of agent distributions
    println!("Part 1: Measuring agent decision entropy");
    let fair_agent = vec![0.25, 0.25, 0.25, 0.25]; // equally distributed
    let biased_agent = vec![0.7, 0.1, 0.1, 0.1];   // favors option A
    
    let h_fair = entropy_conservation::shannon_entropy(&fair_agent);
    let h_biased = entropy_conservation::shannon_entropy(&biased_agent);
    println!("  Fair agent entropy:  {:.3} bits (max: {:.3})", h_fair, entropy_conservation::max_shannon(4));
    println!("  Biased agent entropy: {:.3} bits", h_biased);
    println!();

    // Part 2: KL divergence — how different are two agents?
    println!("Part 2: KL divergence between agent policies");
    let kl = entropy_conservation::kl_divergence(&biased_agent, &fair_agent);
    let js = entropy_conservation::js_divergence(&biased_agent, &fair_agent);
    println!("  KL(biased || fair): {:.4}", kl);
    println!("  JS(biased, fair):   {:.4} (symmetric)", js);
    println!();

    // Part 3: Rényi entropy (generalized family)
    println!("Part 3: Rényi entropy family");
    for alpha in [0.5, 1.0, 2.0, 5.0, 100.0] {
        if let Ok(h) = entropy_conservation::renyi_entropy(&fair_agent, alpha) {
            println!("  H_{:.1}(fair) = {:.3}", alpha, h);
        }
    }
    println!();

    // Part 4: Tsallis entropy (non-extensive)
    println!("Part 4: Tsallis entropy (non-extensive)");
    for q in [0.5, 1.5, 2.0] {
        let s = entropy_conservation::tsallis_entropy(&fair_agent, q);
        println!("  S_q={:.1}(fair) = {:.3}", q, s);
    }
    println!();

    // Part 5: Conservation check — before/after agent operation
    println!("Part 5: Conservation check");
    let before = vec![0.25, 0.25, 0.25, 0.25];
    let after  = vec![0.24, 0.26, 0.25, 0.25]; // small perturbation
    
    match entropy_conservation::check_conservation(&before, &after) {
        Ok(report) => {
            println!("  Conserved: {}", report.is_conserved());
            println!("  Total violation: {:.6}", report.total_violation());
        }
        Err(e) => println!("  Error: {}", e),
    }
}
