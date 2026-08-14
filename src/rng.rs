//! A tiny random number generator.
//!
//! Not part of the neural network at all — we just need *some* source of
//! randomness to initialise weights, and this project has no dependencies.
//! This is xorshift64*: fast, decent statistical quality, about ten lines.
//! Do not use it for cryptography.

pub struct Rng {
    state: u64,
}

impl Rng {
    /// `seed` may be anything except 0 (the generator would get stuck there),
    /// so we force the low bit on.
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A float in [0, 1).
    ///
    /// f64 has 53 bits of mantissa, so we take the top 53 bits of the u64 and
    /// divide by 2^53 to land in the unit interval without losing precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A float in [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stays_in_range() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            let y = rng.uniform(-2.0, 5.0);
            assert!((-2.0..5.0).contains(&y));
        }
    }

    /// Reproducibility is what actually matters here: when a training run
    /// misbehaves you want to replay the exact same initial weights.
    #[test]
    fn same_seed_gives_same_sequence() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        for _ in 0..10 {
            assert_eq!(a.next_f64(), b.next_f64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_f64(), b.next_f64());
    }

    #[test]
    fn is_roughly_uniform() {
        let mut rng = Rng::new(99);
        let n = 10_000;
        let mean: f64 = (0..n).map(|_| rng.next_f64()).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.02, "mean was {mean}");
    }
}
