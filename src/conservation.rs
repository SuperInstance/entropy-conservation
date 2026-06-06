//! Conservation law enforcement: track H over code changes,
//! detect entropy-increasing commits, compute entropy gradient.

use crate::entropy::VerificationEntropy;

/// A single violation of the conservation law dH/dt ≤ 0.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntropyViolation {
    /// Description of what changed.
    pub description: String,
    /// Entropy before the change.
    pub h_before: f64,
    /// Entropy after the change.
    pub h_after: f64,
    /// Magnitude of increase.
    pub delta: f64,
}

/// Report on conservation-law compliance for a sequence of changes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConservationReport {
    /// Shannon entropy before changes.
    pub before: f64,
    /// Shannon entropy after changes.
    pub after: f64,
    /// Net change in entropy.
    pub delta: f64,
    /// Individual violations (entropy increases).
    pub violations: Vec<EntropyViolation>,
}

impl ConservationReport {
    /// Build a conservation report from a sequence of entropy snapshots.
    ///
    /// Each pair of consecutive values represents a change. Any increase
    /// is flagged as a violation.
    pub fn from_snapshots(snapshots: &[VerificationEntropy]) -> Self {
        if snapshots.is_empty() {
            return Self {
                before: 0.0,
                after: 0.0,
                delta: 0.0,
                violations: vec![],
            };
        }

        let before = snapshots[0].shannon;
        let after = snapshots.last().unwrap().shannon;
        let delta = after - before;

        let mut violations = Vec::new();
        for window in snapshots.windows(2) {
            let h_before = window[0].shannon;
            let h_after = window[1].shannon;
            let d = h_after - h_before;
            if d > 1e-12 {
                violations.push(EntropyViolation {
                    description: format!("entropy increased by {:.6} bits", d),
                    h_before,
                    h_after,
                    delta: d,
                });
            }
        }

        Self {
            before,
            after,
            delta,
            violations,
        }
    }

    /// Whether the overall change obeys the conservation law.
    pub fn is_conserved(&self) -> bool {
        self.delta <= 1e-12
    }

    /// Severity: total magnitude of all violations.
    pub fn total_violation(&self) -> f64 {
        self.violations.iter().map(|v| v.delta).sum()
    }
}

/// Given a "before" and "after" probability distribution, check conservation.
pub fn check_conservation(p_before: &[f64], p_after: &[f64]) -> Result<ConservationReport, crate::EntropyError> {
    let ve_before = VerificationEntropy::from_probabilities(p_before)?;
    let ve_after = VerificationEntropy::from_probabilities(p_after)?;
    Ok(ConservationReport::from_snapshots(&[ve_before, ve_after]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ve(shannon: f64) -> VerificationEntropy {
        VerificationEntropy {
            shannon,
            renyi: Default::default(),
            tsallis: 0.0,
        }
    }

    #[test]
    fn decreasing_entropy_is_conserved() {
        let snapshots = vec![make_ve(2.0), make_ve(1.8), make_ve(1.5), make_ve(1.2)];
        let report = ConservationReport::from_snapshots(&snapshots);
        assert!(report.is_conserved());
        assert!(report.violations.is_empty());
    }

    #[test]
    fn increasing_entropy_flags_violations() {
        let snapshots = vec![make_ve(1.0), make_ve(1.5), make_ve(1.2), make_ve(2.0)];
        let report = ConservationReport::from_snapshots(&snapshots);
        assert!(!report.is_conserved());
        assert!(!report.violations.is_empty());
        assert_eq!(report.violations.len(), 2, "two increases in the sequence");
    }

    #[test]
    fn net_decrease_with_local_increases() {
        let snapshots = vec![make_ve(2.0), make_ve(2.3), make_ve(1.8)];
        let report = ConservationReport::from_snapshots(&snapshots);
        // Net decrease but one local violation
        assert!(report.is_conserved(), "net delta is negative");
        assert_eq!(report.violations.len(), 1);
    }

    #[test]
    fn check_conservation_api() {
        let before = vec![0.25, 0.25, 0.25, 0.25]; // H = 2.0
        let after = vec![0.5, 0.3, 0.1, 0.1];      // H < 2.0
        let report = check_conservation(&before, &after).unwrap();
        assert!(report.is_conserved());
    }
}
