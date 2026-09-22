//! The weights blob: the mirror image of `tiny_g2p/export.py`'s writer.
//!
//! Deliberately hand-rolled and dependency-free. The layout is fixed and
//! versioned; `tiny_g2p/tests/test_export.py::Reader` is the same reader in
//! Python, and the two must agree field for field.

use std::collections::HashMap;

/// File magic: the first four bytes of every blob.
pub const MAGIC: &[u8; 4] = b"TG2P";
/// Blob layout version this build reads. `parse` rejects any other value
/// rather than misreading a newer layout.
pub const VERSION: u32 = 1;

/// Why a weights blob could not be loaded.
#[derive(Debug)]
pub enum BlobError {
    /// The first four bytes are not [`MAGIC`].
    BadMagic,
    /// A layout version other than [`VERSION`]; carries the version found.
    UnsupportedVersion(u32),
    /// The data ended in the middle of a section.
    Truncated,
    /// A length-prefixed string was not valid UTF-8.
    BadUtf8,
}

impl std::fmt::Display for BlobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlobError::BadMagic => write!(f, "not a tiny-g2p blob (bad magic)"),
            BlobError::UnsupportedVersion(v) => {
                write!(f, "blob version {v} is newer than this build ({VERSION})")
            }
            BlobError::Truncated => write!(f, "blob ended early"),
            BlobError::BadUtf8 => write!(f, "blob has invalid utf-8 in a string"),
        }
    }
}

impl std::error::Error for BlobError {}

/// A float matrix, row-major, `rows` x `cols`.
///
/// Crate-internal: `row` indexes without a bounds check, so it must not be a
/// public promise. Nothing outside needs a matrix, only [`Blob`] itself.
#[derive(Debug, Clone)]
pub(crate) struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// One quantized linear layer: per-output-channel weight scales, the activation
/// scale it expects, and the scale/zero-point it produces.
#[derive(Debug, Clone)]
pub(crate) struct QuantLinear {
    pub rows: usize,
    pub cols: usize,
    pub relu: bool,
    pub weight: Vec<i8>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
    pub bias: Vec<f32>,
    pub out_scale: f32,
    pub out_zero_point: i32,
}

#[derive(Debug, Clone)]
pub(crate) struct FloatWeights {
    pub embedding: Vec<f32>, // n_src * embed_dim
    pub mlp0: Matrix,
    pub mlp0_bias: Vec<f32>,
    pub mlp2: Matrix,
    pub mlp2_bias: Vec<f32>,
    pub residual: Matrix,
    pub residual_bias: Vec<f32>,
    pub head: Matrix,
    pub head_bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct Int8Weights {
    pub in_scale: f32,
    pub in_zero_point: i32,
    /// mlp0, mlp2, residual, then the phone head (no ReLU).
    pub layers: Vec<QuantLinear>,
    pub add_out_scale: f32,
    pub add_out_zero_point: i32,
}

/// Everything the runtime needs, one struct.
#[derive(Debug, Clone)]
pub struct Blob {
    /// Source vocabulary size: one id per input character, PAD and UNK
    /// included.
    pub n_src: usize,
    /// Target vocabulary size: one id per output phone, PAD, UNK and BLANK
    /// included.
    pub n_tgt: usize,
    /// Number of articulatory manner classes; the label space of the
    /// auxiliary manner head. Carried because the layout mirrors the
    /// exporter, but the runtime never reads it.
    pub n_manners: usize,
    /// Width of one character embedding row.
    pub embed_dim: usize,
    /// Width of the first MLP layer.
    pub hidden_dim: usize,
    /// Width of the context vector the residual sum and the phone head share.
    pub ctx_dim: usize,
    /// Character offsets gathered around each position, e.g. `-2` reads two
    /// characters to the left. Fixed offsets, so edge positions read PAD.
    pub taps: Vec<i32>,
    /// Input characters by id: `src_itos[id]` is the character for `id`, with
    /// `pad_id` and `unk_id` the reserved entries.
    pub src_itos: Vec<String>,
    /// Output phones by id, with PAD, UNK and BLANK reserved; `blank_id` is
    /// the label for a character that emits no phone.
    pub tgt_itos: Vec<String>,
    /// Manner-class names by id, the label space of the auxiliary manner
    /// head; carried but unused at inference.
    pub manner_itos: Vec<String>,
    /// PAD id: what the taps read past a word's edge, whose embedding row is
    /// trained to zero.
    pub pad_id: usize,
    /// UNK id: the character id for input the vocabulary does not contain.
    pub unk_id: usize,
    /// BLANK id: the phone label for a character that emits no phone.
    pub blank_id: usize,
    /// Letters whose merged nasal label may be split back (ą, ę).
    pub nasal_letters: Vec<String>,
    /// Merged nasal vowel -> its oral half (`ɔ̃` -> `ɔ`, `ɛ̃` -> `ɛ`), so the
    /// split can restore MFA's oral+nasal form.
    pub oral: HashMap<String, String>,
    /// Raw MFA stops and affricates. A merged nasal vowel followed by one of
    /// these is always an ą/ę merge and is split.
    pub stops: Vec<String>,
    /// Follower stop -> the homorganic nasal MFA writes after the split
    /// (`t̪` -> `n̪`, `k` -> `ŋ`).
    pub homorganic: HashMap<String, String>,
    /// Polish letter names, for spelling initialisms.
    pub letter_names: HashMap<String, String>,
    /// Words we hold a gold letter-by-letter reading for.
    pub acronyms: HashMap<String, String>,
    pub(crate) float: FloatWeights,
    pub(crate) int8: Int8Weights,
}

impl Blob {
    /// Parse a blob. Every section is required; a half-exported blob is an
    /// error rather than a silently smaller model.
    pub fn parse(data: &[u8]) -> Result<Blob, BlobError> {
        let mut r = Reader::new(data);
        if r.raw(4)? != MAGIC {
            return Err(BlobError::BadMagic);
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(BlobError::UnsupportedVersion(version));
        }
        let n_src = r.u32()? as usize;
        let n_tgt = r.u32()? as usize;
        let n_manners = r.u32()? as usize;
        let embed_dim = r.u32()? as usize;
        let hidden_dim = r.u32()? as usize;
        let ctx_dim = r.u32()? as usize;
        let n_taps = r.u32()? as usize;
        let taps = r.i32s(n_taps)?;
        let src_itos = r.strings()?;
        let tgt_itos = r.strings()?;
        let manner_itos = r.strings()?;
        let pad_id = r.u32()? as usize;
        let unk_id = r.u32()? as usize;
        let blank_id = r.u32()? as usize;

        let nasal_letters = r.strings()?;
        let oral = r.pairs()?;
        let stops = r.strings()?;
        let homorganic = r.pairs()?;
        let letter_names = r.pairs()?;
        let acronyms = r.pairs()?;

        if r.u8()? != 1 {
            return Err(BlobError::Truncated);
        }
        let float = FloatWeights {
            embedding: r.f32s(n_src * embed_dim)?,
            mlp0: read_matrix(&mut r, hidden_dim, embed_dim * n_taps)?,
            mlp0_bias: r.f32s(hidden_dim)?,
            mlp2: read_matrix(&mut r, ctx_dim, hidden_dim)?,
            mlp2_bias: r.f32s(ctx_dim)?,
            residual: read_matrix(&mut r, ctx_dim, ctx_dim)?,
            residual_bias: r.f32s(ctx_dim)?,
            head: read_matrix(&mut r, n_tgt, ctx_dim)?,
            head_bias: r.f32s(n_tgt)?,
        };

        if r.u8()? != 1 {
            return Err(BlobError::Truncated);
        }
        let in_scale = r.f32()?;
        let in_zero_point = r.i32()?;
        let mut layers = Vec::with_capacity(4);
        for _ in 0..4 {
            let rows = r.u32()? as usize;
            let cols = r.u32()? as usize;
            let relu = r.u8()? == 1;
            layers.push(QuantLinear {
                rows,
                cols,
                relu,
                weight: r.i8s(rows * cols)?,
                scales: r.f32s(rows)?,
                zero_points: r.i32s(rows)?,
                bias: r.f32s(rows)?,
                out_scale: r.f32()?,
                out_zero_point: r.i32()?,
            });
        }
        let add_out_scale = r.f32()?;
        let add_out_zero_point = r.i32()?;

        Ok(Blob {
            n_src,
            n_tgt,
            n_manners,
            embed_dim,
            hidden_dim,
            ctx_dim,
            taps,
            src_itos,
            tgt_itos,
            manner_itos,
            pad_id,
            unk_id,
            blank_id,
            nasal_letters,
            oral,
            stops,
            homorganic,
            letter_names,
            acronyms,
            float,
            int8: Int8Weights {
                in_scale,
                in_zero_point,
                layers,
                add_out_scale,
                add_out_zero_point,
            },
        })
    }

    /// The character window the taps cover: `2 * max|tap| + 1`. Words longer
    /// than this still transcribe -- the model extrapolates.
    pub fn window_len(&self) -> usize {
        let reach = self.taps.iter().map(|t| t.unsigned_abs() as usize).max().unwrap_or(0);
        2 * reach + 1
    }

    /// Embedding row for a token id, zeroed for PAD (the model trains that row
    /// to zero and the taps rely on it).
    #[inline]
    pub fn embed_row(&self, id: usize) -> &[f32] {
        let dim = self.embed_dim;
        &self.float.embedding[id * dim..(id + 1) * dim]
    }
}

fn read_matrix(r: &mut Reader, rows: usize, cols: usize) -> Result<Matrix, BlobError> {
    Ok(Matrix { rows, cols, data: r.f32s(rows * cols)? })
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn raw(&mut self, n: usize) -> Result<&'a [u8], BlobError> {
        if self.pos + n > self.data.len() {
            return Err(BlobError::Truncated);
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, BlobError> {
        Ok(self.raw(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, BlobError> {
        let b = self.raw(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32, BlobError> {
        Ok(self.u32()? as i32)
    }

    fn f32(&mut self) -> Result<f32, BlobError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, BlobError> {
        (0..n).map(|_| self.f32()).collect()
    }

    fn i8s(&mut self, n: usize) -> Result<Vec<i8>, BlobError> {
        Ok(self.raw(n)?.iter().map(|b| *b as i8).collect())
    }

    fn i32s(&mut self, n: usize) -> Result<Vec<i32>, BlobError> {
        (0..n).map(|_| self.i32()).collect()
    }

    fn string(&mut self) -> Result<String, BlobError> {
        let b = self.raw(2)?;
        let len = u16::from_le_bytes([b[0], b[1]]) as usize;
        let bytes = self.raw(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| BlobError::BadUtf8)
    }

    fn strings(&mut self) -> Result<Vec<String>, BlobError> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.string()).collect()
    }

    fn pairs(&mut self) -> Result<HashMap<String, String>, BlobError> {
        let n = self.u32()? as usize;
        let mut out = HashMap::with_capacity(n);
        for _ in 0..n {
            let key = self.string()?;
            let value = self.string()?;
            out.insert(key, value);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_blob_parses() {
        let blob = Blob::parse(crate::EMBEDDED_WEIGHTS).expect("embedded blob");
        assert_eq!(blob.n_src, 37);
        assert_eq!(blob.n_tgt, 53);
        assert_eq!(blob.taps.len(), 11);
        assert_eq!(blob.int8.layers.len(), 4);
        assert!(!blob.int8.layers[3].relu, "the phone head is un-activated");
        assert_eq!(blob.oral.get("ɔ\u{303}").map(String::as_str), Some("ɔ"));
    }

    #[test]
    fn a_bad_magic_is_rejected() {
        let mut data = crate::EMBEDDED_WEIGHTS.to_vec();
        data[0] = b'X';
        assert!(matches!(Blob::parse(&data), Err(BlobError::BadMagic)));
    }

    #[test]
    fn a_newer_version_is_rejected_before_it_is_misread() {
        let mut data = crate::EMBEDDED_WEIGHTS.to_vec();
        data[4] = 99;
        match Blob::parse(&data) {
            Err(BlobError::UnsupportedVersion(99)) => {}
            other => panic!("expected a version error, got {other:?}"),
        }
    }

    #[test]
    fn a_truncated_blob_is_rejected() {
        let half = &crate::EMBEDDED_WEIGHTS[..crate::EMBEDDED_WEIGHTS.len() / 2];
        assert!(matches!(Blob::parse(half), Err(BlobError::Truncated)));
    }
}
