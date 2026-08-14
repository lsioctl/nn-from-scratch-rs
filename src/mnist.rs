//! Step 4: reading real data.
//!
//! MNIST is 70,000 handwritten digits, 28x28 pixels, greyscale — 60,000 for
//! training and 10,000 held back for testing. It is stored in "IDX" format,
//! which is refreshingly simple: a short big-endian header, then raw bytes.
//!
//!   images file            labels file
//!   ------------------     ------------------
//!   magic  0x00000803      magic  0x00000801
//!   count  60000           count  60000
//!   rows   28              u8 label per image
//!   cols   28
//!   u8 pixel per byte
//!
//! The magic number encodes the element type (0x08 = unsigned byte) and the
//! number of dimensions (3 for images, 1 for labels), which is why the two
//! differ in the last digit.

use std::fs;
use std::io;
use std::path::Path;

/// A dataset of images and their digit labels.
pub struct MnistData {
    /// One entry per image: 784 pixels, each already scaled to [0, 1].
    pub images: Vec<Vec<f64>>,
    /// The true digit, 0-9.
    pub labels: Vec<u8>,
}

impl MnistData {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Convert to the `(inputs, targets)` pairs our `Network` trains on.
    ///
    /// The label is expanded into a **one-hot** vector: the digit 3 becomes
    /// `[0,0,0,1,0,0,0,0,0,0]`. We need this because the network has ten
    /// output neurons, and each needs its own target. Feeding the raw number
    /// 3 to a single output would be much worse — it would imply 4 is "closer
    /// to correct" than 8 when the true answer is 3, and digits have no such
    /// ordering.
    pub fn to_samples(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        self.images
            .iter()
            .zip(&self.labels)
            .map(|(image, &label)| {
                let mut target = vec![0.0; 10];
                target[label as usize] = 1.0;
                (image.clone(), target)
            })
            .collect()
    }
}

/// Read a big-endian u32 from a byte slice.
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Parse an IDX image file into rows of pixels scaled to [0, 1].
fn load_images(path: &Path) -> io::Result<Vec<Vec<f64>>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 16 {
        return Err(invalid(format!("{}: too short to be an IDX file", path.display())));
    }

    let magic = read_u32(&bytes, 0);
    if magic != 0x0000_0803 {
        return Err(invalid(format!(
            "{}: expected image magic 0x00000803, found {magic:#010x}",
            path.display()
        )));
    }

    let count = read_u32(&bytes, 4) as usize;
    let rows = read_u32(&bytes, 8) as usize;
    let cols = read_u32(&bytes, 12) as usize;
    let pixels_per_image = rows * cols;

    let expected = 16 + count * pixels_per_image;
    if bytes.len() != expected {
        return Err(invalid(format!(
            "{}: expected {expected} bytes for {count} {rows}x{cols} images, found {}",
            path.display(),
            bytes.len()
        )));
    }

    // Scale bytes from 0..=255 down to 0.0..=1.0.
    //
    // This is not cosmetic. Raw pixel values up to 255 would produce enormous
    // weighted sums, pinning every sigmoid at 0 or 1 where its derivative is
    // ~0 — the saturation problem — and the network would refuse to learn.
    // Getting inputs into a sane range is the most basic form of what is
    // generally called feature scaling, and skipping it is a classic way to
    // waste an afternoon.
    Ok(bytes[16..]
        .chunks_exact(pixels_per_image)
        .map(|image| image.iter().map(|&p| p as f64 / 255.0).collect())
        .collect())
}

/// Parse an IDX label file.
fn load_labels(path: &Path) -> io::Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 8 {
        return Err(invalid(format!("{}: too short to be an IDX file", path.display())));
    }

    let magic = read_u32(&bytes, 0);
    if magic != 0x0000_0801 {
        return Err(invalid(format!(
            "{}: expected label magic 0x00000801, found {magic:#010x}",
            path.display()
        )));
    }

    let count = read_u32(&bytes, 4) as usize;
    let expected = 8 + count;
    if bytes.len() != expected {
        return Err(invalid(format!(
            "{}: expected {expected} bytes for {count} labels, found {}",
            path.display(),
            bytes.len()
        )));
    }

    let labels = bytes[8..].to_vec();
    if let Some(&bad) = labels.iter().find(|&&l| l > 9) {
        return Err(invalid(format!("{}: label {bad} is not a digit", path.display())));
    }

    Ok(labels)
}

/// Load one split ("train" or "t10k") from a directory of IDX files.
fn load_split(dir: &Path, images_file: &str, labels_file: &str) -> io::Result<MnistData> {
    let images = load_images(&dir.join(images_file))?;
    let labels = load_labels(&dir.join(labels_file))?;

    if images.len() != labels.len() {
        return Err(invalid(format!(
            "{images_file} has {} images but {labels_file} has {} labels",
            images.len(),
            labels.len()
        )));
    }

    Ok(MnistData { images, labels })
}

/// Load the 60,000 training digits.
pub fn load_training(dir: impl AsRef<Path>) -> io::Result<MnistData> {
    load_split(
        dir.as_ref(),
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    )
}

/// Load the 10,000 test digits.
///
/// These are kept strictly separate. The whole point of machine learning is
/// performing well on data you have never seen, and the only way to know
/// whether you do is to never train on this split.
pub fn load_test(dir: impl AsRef<Path>) -> io::Result<MnistData> {
    load_split(
        dir.as_ref(),
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
}

/// Render a digit as ASCII art, for sanity-checking that we parsed correctly.
pub fn render(image: &[f64]) -> String {
    const SHADES: [char; 5] = [' ', '.', '+', '*', '#'];

    (0..28)
        .map(|row| {
            (0..28)
                .map(|col| {
                    let value = image[row * 28 + col];
                    let level = ((value * (SHADES.len() - 1) as f64).round() as usize)
                        .min(SHADES.len() - 1);
                    SHADES[level]
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
            .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR: &str = "data";

    /// These tests need the dataset; skip cleanly if `fetch-mnist.sh` has not
    /// been run, so `cargo test` still works on a fresh clone.
    fn training_or_skip() -> Option<MnistData> {
        load_training(DIR).ok()
    }

    #[test]
    fn training_split_has_the_expected_shape() {
        let Some(data) = training_or_skip() else {
            eprintln!("skipping: run ./fetch-mnist.sh first");
            return;
        };

        assert_eq!(data.len(), 60_000);
        assert_eq!(data.images[0].len(), 784);
        assert!(data.images.iter().all(|i| i.len() == 784));
    }

    #[test]
    fn test_split_has_the_expected_shape() {
        let Ok(data) = load_test(DIR) else {
            eprintln!("skipping: run ./fetch-mnist.sh first");
            return;
        };

        assert_eq!(data.len(), 10_000);
        assert!(data.labels.iter().all(|&l| l <= 9));
    }

    #[test]
    fn pixels_are_scaled_into_the_unit_interval() {
        let Some(data) = training_or_skip() else { return };

        let all = data.images.iter().flatten();
        assert!(all.clone().all(|&p| (0.0..=1.0).contains(&p)));
        // A real digit image must contain both ink and blank paper.
        assert!(data.images[0].iter().any(|&p| p == 0.0));
        assert!(data.images[0].iter().any(|&p| p > 0.9));
    }

    #[test]
    fn one_hot_targets_have_exactly_one_hot_entry() {
        let Some(data) = training_or_skip() else { return };

        let samples = data.to_samples();
        let (_, target) = &samples[0];
        assert_eq!(target.len(), 10);
        assert_eq!(target.iter().sum::<f64>(), 1.0);
        assert_eq!(target[data.labels[0] as usize], 1.0);
    }

    /// Every digit should appear a few thousand times; a parsing slip that
    /// shifted the label stream would show up as a wildly skewed histogram.
    #[test]
    fn all_ten_digits_are_present_in_sane_proportions() {
        let Some(data) = training_or_skip() else { return };

        let mut counts = [0usize; 10];
        for &l in &data.labels {
            counts[l as usize] += 1;
        }
        assert!(
            counts.iter().all(|&c| (4_000..8_000).contains(&c)),
            "suspicious label distribution: {counts:?}"
        );
    }
}
