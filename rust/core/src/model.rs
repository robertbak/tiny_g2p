//! The forward pass, in two modes over the same blob.
//!
//! The graph is `embed -> gather 11 taps -> MLP -> +residual -> phone head`;
//! no recurrence and no attention, so this is a couple of hundred lines of
//! arithmetic that runs anywhere.
//!
//! **Float** (the default) uses the trained checkpoint's own weights. On the
//! held-out set it scores the same word accuracy as the int8 artifact
//! (97.89% either way, differing on 23 of 6,719 words), so float costs
//! nothing in fidelity and is the simpler path.
//!
//! **Int8** replays the quantized artifact exactly: per-output-channel weight
//! scales with zero offsets, `x_zero_point`-shifted integer accumulation, and
//! the activation scales the exported observers froze. `--int8` exists so a
//! deployment can *prove* it reproduces the measured numbers, not to save
//! space -- the blob is 142 KiB either way.

use crate::blob::{Blob, Matrix, QuantLinear};

/// The training window: `2 * max|tap| + 1` = 33 characters, which covers the
/// longest word in the lexicon (32). It is **not** a limit on input -- the
/// taps are fixed offsets, so a longer word still computes, it is simply
/// extrapolating beyond anything the model saw. Ask [`Blob::window_len`] for
/// the value this build actually carries; this crate transcribes and leaves
/// the judgement to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The trained checkpoint's own weights: the default, and the simpler
    /// arithmetic.
    Float,
    /// Replay the quantized int8 artifact exactly, so a deployment can prove
    /// it reproduces the measured numbers.
    Int8,
}

/// A quantized activation: bytes plus the affine map that gives them meaning.
#[derive(Debug, Clone)]
struct QTensor {
    data: Vec<u8>,
    scale: f32,
    zero_point: i32,
}

pub struct Model<'a> {
    blob: &'a Blob,
    mode: Mode,
}

impl<'a> Model<'a> {
    pub fn new(blob: &'a Blob, mode: Mode) -> Self {
        Model { blob, mode }
    }

    /// Per-character argmax over the phone head, in token ids. Infallible:
    /// any length computes, in any mode, from any vocabulary.
    ///
    /// Blank/PAD/UNK filtering and the nasal split happen in the caller
    /// (`text::decode` and `phones::restore`), so this stays pure arithmetic.
    pub fn argmax(&self, ids: &[usize]) -> Vec<usize> {
        match self.mode {
            Mode::Float => self.forward_float(ids),
            Mode::Int8 => self.forward_int8(ids),
        }
    }

    /// The gathered tap vectors: one row of `embed_dim * n_taps` per character,
    /// padded so out-of-range taps read PAD's zero embedding row.
    ///
    /// Layout matters: the Python side builds `[B, T, E, taps]` with
    /// `torch.stack(..., dim=-1)` and then reshapes, so the flattened index is
    /// `d * n_taps + t` -- embedding dimension first, taps second. Getting this
    /// backwards silently transposes the first layer's input.
    fn gather_taps(&self, ids: &[usize]) -> Vec<f32> {
        let blob = self.blob;
        let dim = blob.embed_dim;
        let n_taps = blob.taps.len();
        let len = ids.len();
        let mut flat = vec![0.0f32; len * dim * n_taps];
        for (t, &offset) in blob.taps.iter().enumerate() {
            for pos in 0..len {
                let src = pos as i32 + offset;
                if src < 0 || src as usize >= len {
                    continue; // stays zero: PAD's row
                }
                let row = blob.embed_row(ids[src as usize]);
                let base = pos * dim * n_taps + t;
                for (d, value) in row.iter().enumerate() {
                    flat[base + d * n_taps] = *value;
                }
            }
        }
        flat
    }

    fn forward_float(&self, ids: &[usize]) -> Vec<usize> {
        let f = &self.blob.float;
        let len = ids.len();
        let flat = self.gather_taps(ids);
        let h1 = linear(&f.mlp0, &f.mlp0_bias, &flat, len, true);
        let ctx = linear(&f.mlp2, &f.mlp2_bias, &h1, len, true);
        let res = linear(&f.residual, &f.residual_bias, &ctx, len, true);
        let summed: Vec<f32> = ctx.iter().zip(&res).map(|(a, b)| a + b).collect();
        // The head is un-activated: argmax over its raw logits.
        let logits = linear(&f.head, &f.head_bias, &summed, len, false);
        argmax_rows(&logits, len, self.blob.n_tgt)
    }

    fn forward_int8(&self, ids: &[usize]) -> Vec<usize> {
        let q = &self.blob.int8;
        let len = ids.len();
        let flat = self.gather_taps(ids);
        let dim = flat.len() / len.max(1);
        let mut out = Vec::with_capacity(len);
        for pos in 0..len {
            let row = &flat[pos * dim..(pos + 1) * dim];
            let x = QTensor {
                data: quantize(row, q.in_scale, q.in_zero_point),
                scale: q.in_scale,
                zero_point: q.in_zero_point,
            };
            let h1 = quant_linear(&q.layers[0], &x);
            let ctx = quant_linear(&q.layers[1], &h1);
            let res = quant_linear(&q.layers[2], &ctx);
            let summed = quant_add(&ctx, &res, q.add_out_scale, q.add_out_zero_point);
            let logits = quant_linear(&q.layers[3], &summed);
            // Dequantizing is a positive affine map, so the argmax over the
            // clamped bytes is the argmax torch computes on floats.
            out.push(argmax(&logits.data));
        }
        out
    }
}

// -- float helpers --------------------------------------------------------

/// `W x + b` for `n` rows of `x`, optionally ReLU'd.
#[inline]
fn linear(matrix: &Matrix, bias: &[f32], x: &[f32], n: usize, relu: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; n * matrix.rows];
    for p in 0..n {
        let xp = &x[p * matrix.cols..(p + 1) * matrix.cols];
        let dst = &mut out[p * matrix.rows..(p + 1) * matrix.rows];
        for (r, row) in matrix.data.chunks_exact(matrix.cols).enumerate() {
            let mut acc = bias[r];
            for (w, v) in row.iter().zip(xp) {
                acc += w * v;
            }
            dst[r] = if relu && acc < 0.0 { 0.0 } else { acc };
        }
    }
    out
}

#[inline]
fn argmax(values: &[u8]) -> usize {
    let mut best = 0usize;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

#[inline]
fn argmax_rows(values: &[f32], n: usize, rows: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    for p in 0..n {
        let row = &values[p * rows..(p + 1) * rows];
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        out.push(best);
    }
    out
}

// -- int8 helpers ---------------------------------------------------------

#[inline]
fn quantize(x: &[f32], scale: f32, zero_point: i32) -> Vec<u8> {
    x.iter()
        .map(|v| {
            let q = (*v as f64 / scale as f64).round() + zero_point as f64;
            clamp_u8(q)
        })
        .collect()
}

/// One quantized linear: integer accumulation, then requantize into the
/// layer's own output scale, then (optionally) clamp at the output zero point
/// -- which is what `torch.relu` does to a quantized tensor.
fn quant_linear(layer: &QuantLinear, x: &QTensor) -> QTensor {
    let mut out = vec![0u8; layer.rows];
    for (r, slot) in out.iter_mut().enumerate() {
        let w = &layer.weight[r * layer.cols..(r + 1) * layer.cols];
        let mut acc: i32 = 0;
        for (i, wq) in w.iter().enumerate() {
            let shifted_w = *wq as i32 - layer.zero_points[r];
            let shifted_x = x.data[i] as i32 - x.zero_point;
            acc += shifted_w * shifted_x;
        }
        debug_assert!(acc >= 0 || layer.zero_points[r] != 0);
        let multiplier = x.scale as f64 * layer.scales[r] as f64 / layer.out_scale as f64;
        let value = acc as f64 * multiplier + layer.bias[r] as f64 / layer.out_scale as f64;
        let mut byte = clamp_u8(value.round() + layer.out_zero_point as f64);
        if layer.relu {
            let floor = layer.out_zero_point.clamp(0, 255) as u8;
            if byte < floor {
                byte = floor;
            }
        }
        *slot = byte;
    }
    QTensor { data: out, scale: layer.out_scale, zero_point: layer.out_zero_point }
}

/// `a + b` in the float domain, requantized into the add's own scale. The
/// graph's residual sum is un-activated -- torch's convert inserts no ReLU
/// after the FloatFunctional -- so this deliberately clamps nothing but the
/// byte range.
fn quant_add(a: &QTensor, b: &QTensor, out_scale: f32, out_zero_point: i32) -> QTensor {
    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(av, bv)| {
            let value = a.scale as f64 * (*av as i32 - a.zero_point) as f64
                + b.scale as f64 * (*bv as i32 - b.zero_point) as f64;
            clamp_u8((value / out_scale as f64).round() + out_zero_point as f64)
        })
        .collect();
    QTensor { data, scale: out_scale, zero_point: out_zero_point }
}

#[inline]
fn clamp_u8(value: f64) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value as u8
    }
}
