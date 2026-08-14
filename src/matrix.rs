//! Step 5: a matrix, and why the whole network wants to be one.
//!
//! Until now a layer looped over its neurons, and training looped over its
//! samples, and each neuron looped over its weights. Three nested loops of
//! `Vec<f64>` allocations, chasing pointers all over the heap.
//!
//! Rewriting this as matrix arithmetic buys two things:
//!
//!   * **Speed.** One flat `Vec<f64>` in row-major order means the inner loop
//!     walks contiguous memory, which is what caches and SIMD units want.
//!   * **Clarity.** The entire forward pass for a batch of 32 images becomes
//!     `X * W + b`. The backward pass becomes three more products. All the
//!     bookkeeping loops disappear.
//!
//! Everything here is stored **row-major**: `data[row * cols + col]`.

/// A dense 2-D matrix of f64, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "{rows}x{cols} needs {} values, got {}",
            rows * cols,
            data.len()
        );
        Self { rows, cols, data }
    }

    /// Build from a list of rows — handy in tests and when loading data.
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        assert!(!rows.is_empty(), "need at least one row");
        let cols = rows[0].len();
        assert!(
            rows.iter().all(|r| r.len() == cols),
            "all rows must have the same length"
        );
        Self {
            rows: rows.len(),
            cols,
            data: rows.concat(),
        }
    }

    /// A single sample as a 1-row matrix.
    pub fn row_vector(values: &[f64]) -> Self {
        Self {
            rows: 1,
            cols: values.len(),
            data: values.to_vec(),
        }
    }

    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, value: f64) {
        self.data[r * self.cols + c] = value;
    }

    /// Matrix product: `(m, k) * (k, n) -> (m, n)`.
    ///
    /// The loop order is `i, k, j` rather than the textbook `i, j, k`, and
    /// that choice is most of the performance.
    ///
    /// With `i, j, k` the inner loop reads `rhs[k][j]` for successive `k` —
    /// striding `n` floats through memory every step, missing the cache
    /// constantly. With `i, k, j` the inner loop walks one row of `rhs` and
    /// one row of the output, both contiguous, accumulating `a * b` into
    /// them. Same arithmetic, same result, several times faster, and the
    /// compiler can vectorise it.
    pub fn matmul(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, rhs.rows,
            "cannot multiply {}x{} by {}x{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );

        let mut out = Matrix::zeros(self.rows, rhs.cols);

        for i in 0..self.rows {
            let lhs_row = self.row(i);
            let out_row = &mut out.data[i * rhs.cols..(i + 1) * rhs.cols];

            for (k, &a) in lhs_row.iter().enumerate() {
                // Skipping zeros is a real win on MNIST, where roughly 80% of
                // every image is blank background.
                if a == 0.0 {
                    continue;
                }
                let rhs_row = &rhs.data[k * rhs.cols..(k + 1) * rhs.cols];
                for (o, &b) in out_row.iter_mut().zip(rhs_row) {
                    *o += a * b;
                }
            }
        }

        out
    }

    /// `self^T` — swap rows and columns.
    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
    }

    /// Add a row vector to every row — "broadcasting" a bias across a batch.
    pub fn add_row_broadcast(&mut self, bias: &[f64]) {
        assert_eq!(bias.len(), self.cols, "bias width must match columns");
        for r in 0..self.rows {
            for (value, b) in self.row_mut(r).iter_mut().zip(bias) {
                *value += b;
            }
        }
    }

    /// Sum down each column, giving one value per column.
    ///
    /// This is how a bias gradient is collected: the bias affected every
    /// sample in the batch, so its gradient is the sum over all of them.
    pub fn column_sums(&self) -> Vec<f64> {
        let mut sums = vec![0.0; self.cols];
        for r in 0..self.rows {
            for (s, v) in sums.iter_mut().zip(self.row(r)) {
                *s += v;
            }
        }
        sums
    }

    /// Apply a function to every element, in place.
    pub fn map_in_place(&mut self, f: impl Fn(f64) -> f64) {
        for v in &mut self.data {
            *v = f(*v);
        }
    }

    /// Elementwise multiply, in place — the Hadamard product.
    pub fn multiply_elementwise(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a *= b;
        }
    }

    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.data {
            *v *= factor;
        }
    }

    /// Add another matrix into this one, in place.
    pub fn add_in_place(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    /// Gather the given rows into a new matrix — used to cut a shuffled
    /// minibatch out of the full dataset.
    pub fn select_rows(&self, indices: &[usize]) -> Matrix {
        let mut out = Matrix::zeros(indices.len(), self.cols);
        for (dest, &src) in indices.iter().enumerate() {
            out.row_mut(dest).copy_from_slice(self.row(src));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_a_hand_computation() {
        // [1 2 3]   [ 7  8]     [ 58  64]
        // [4 5 6] * [ 9 10]  =  [139 154]
        //           [11 12]
        let a = Matrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let b = Matrix::from_rows(&[vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);

        let c = a.matmul(&b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn identity_changes_nothing() {
        let a = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let identity = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(a.matmul(&identity), a);
        assert_eq!(identity.matmul(&a), a);
    }

    /// (AB)^T == B^T A^T — catches transpose and indexing mistakes at once.
    #[test]
    fn transpose_of_product_reverses_it() {
        let a = Matrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let b = Matrix::from_rows(&[vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);

        assert_eq!(
            a.matmul(&b).transpose(),
            b.transpose().matmul(&a.transpose())
        );
    }

    #[test]
    fn transpose_is_its_own_inverse() {
        let a = Matrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(a.transpose().transpose(), a);
        assert_eq!(a.transpose().rows, 3);
        assert_eq!(a.transpose().cols, 2);
    }

    #[test]
    #[should_panic(expected = "cannot multiply 2x3 by 2x2")]
    fn mismatched_shapes_are_rejected() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 2);
        a.matmul(&b);
    }

    #[test]
    fn broadcast_adds_to_every_row() {
        let mut a = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        a.add_row_broadcast(&[10.0, 20.0]);
        assert_eq!(a.data, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn column_sums_add_down_the_batch() {
        let a = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(a.column_sums(), vec![9.0, 12.0]);
    }

    #[test]
    fn select_rows_gathers_a_minibatch() {
        let a = Matrix::from_rows(&[
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ]);
        let batch = a.select_rows(&[2, 0]);
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.row(0), &[3.0, 3.0]);
        assert_eq!(batch.row(1), &[1.0, 1.0]);
    }

    #[test]
    fn elementwise_multiply_and_scale() {
        let mut a = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_rows(&[vec![10.0, 100.0], vec![2.0, 0.5]]);
        a.multiply_elementwise(&b);
        assert_eq!(a.data, vec![10.0, 200.0, 6.0, 2.0]);

        a.scale(0.5);
        assert_eq!(a.data, vec![5.0, 100.0, 3.0, 1.0]);
    }

    /// Zero-skipping in matmul must not change the answer.
    #[test]
    fn sparse_inputs_give_the_same_result_as_dense_maths() {
        let sparse = Matrix::from_rows(&[vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 0.0]]);
        let b = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);

        let got = sparse.matmul(&b);
        assert_eq!(got.data, vec![6.0, 8.0, 0.0, 0.0]);
    }
}
