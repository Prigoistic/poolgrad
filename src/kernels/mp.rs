#![allow(dead_code)]

/// MP-inspired block-level bilinear transform.
///
/// This is implemented as a single algebraic transform over 2×2 sub-blocks (Strassen form),
/// and is *not* recursive: the inner multiplies are standard O(n^3) square matrix multiplies.
///
/// The transform works for any even `block_size >= 2` by treating sub-blocks as (block/2)×(block/2)
/// matrices.
#[derive(Debug, Clone)]
pub struct MPTransform {
    /// For each multiplication Mi, a linear combination of A quadrants.
    /// Quadrant indices: 0=A11, 1=A12, 2=A21, 3=A22.
    pub a_terms: Vec<Vec<(usize, f32)>>,
    /// For each multiplication Mi, a linear combination of B quadrants.
    /// Quadrant indices: 0=B11, 1=B12, 2=B21, 3=B22.
    pub b_terms: Vec<Vec<(usize, f32)>>,
    /// For each output quadrant Cq, a linear combination of Mi results.
    /// Output indices: 0=C11, 1=C12, 2=C21, 3=C22.
    pub c_terms: Vec<Vec<(usize, f32)>>,
}

impl MPTransform {
    pub fn strassen() -> Self {
        // M1 = (A11 + A22) (B11 + B22)
        // M2 = (A21 + A22) B11
        // M3 = A11 (B12 - B22)
        // M4 = A22 (B21 - B11)
        // M5 = (A11 + A12) B22
        // M6 = (A21 - A11) (B11 + B12)
        // M7 = (A12 - A22) (B21 + B22)
        //
        // C11 = M1 + M4 - M5 + M7
        // C12 = M3 + M5
        // C21 = M2 + M4
        // C22 = M1 - M2 + M3 + M6

        let a_terms = vec![
            vec![(0, 1.0), (3, 1.0)],  // M1
            vec![(2, 1.0), (3, 1.0)],  // M2
            vec![(0, 1.0)],            // M3
            vec![(3, 1.0)],            // M4
            vec![(0, 1.0), (1, 1.0)],  // M5
            vec![(2, 1.0), (0, -1.0)], // M6
            vec![(1, 1.0), (3, -1.0)], // M7
        ];

        let b_terms = vec![
            vec![(0, 1.0), (3, 1.0)],  // M1
            vec![(0, 1.0)],            // M2
            vec![(1, 1.0), (3, -1.0)], // M3
            vec![(2, 1.0), (0, -1.0)], // M4
            vec![(3, 1.0)],            // M5
            vec![(0, 1.0), (1, 1.0)],  // M6
            vec![(2, 1.0), (3, 1.0)],  // M7
        ];

        let c_terms = vec![
            vec![(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)], // C11
            vec![(2, 1.0), (4, 1.0)],                      // C12
            vec![(1, 1.0), (3, 1.0)],                      // C21
            vec![(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)], // C22
        ];

        Self {
            a_terms,
            b_terms,
            c_terms,
        }
    }

    #[inline]
    pub fn mults(&self) -> usize {
        self.a_terms.len()
    }
}

#[derive(Debug, Default)]
pub struct MPScratch {
    buf: Vec<f32>,
    block: usize,
}

impl MPScratch {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            block: 0,
        }
    }

    pub fn ensure(&mut self, block: usize) {
        if self.block == block && !self.buf.is_empty() {
            return;
        }
        self.block = block;

        let h = block / 2;
        let hh = h * h;
        // Layout: A_quads(4) + B_quads(4) + S + T + M(7) + C_quads(4) = 21 blocks.
        let required = 21 * hh;
        self.buf.resize(required, 0.0);
    }

    fn split_mut(&mut self, block: usize) -> ScratchSlices<'_> {
        self.ensure(block);

        let h = block / 2;
        let hh = h * h;
        let mut offset = 0;

        let (a_quads, rest) = self.buf.split_at_mut(4 * hh);
        offset += 4 * hh;
        let (b_quads, rest) = rest.split_at_mut(4 * hh);
        offset += 4 * hh;
        let (s, rest) = rest.split_at_mut(hh);
        offset += hh;
        let (t, rest) = rest.split_at_mut(hh);
        offset += hh;
        let (m, rest) = rest.split_at_mut(7 * hh);
        offset += 7 * hh;
        let (c_quads, _rest) = rest.split_at_mut(4 * hh);

        debug_assert_eq!(offset + 4 * hh, 21 * hh);

        ScratchSlices {
            h,
            hh,
            a_quads,
            b_quads,
            s,
            t,
            m,
            c_quads,
        }
    }
}

struct ScratchSlices<'a> {
    h: usize,
    hh: usize,
    a_quads: &'a mut [f32],
    b_quads: &'a mut [f32],
    s: &'a mut [f32],
    t: &'a mut [f32],
    m: &'a mut [f32],
    c_quads: &'a mut [f32],
}

#[inline]
fn quad_slice(quads: &[f32], hh: usize, idx: usize) -> &[f32] {
    let start = idx * hh;
    &quads[start..start + hh]
}

#[inline]
fn quad_slice_mut(quads: &mut [f32], hh: usize, idx: usize) -> &mut [f32] {
    let start = idx * hh;
    &mut quads[start..start + hh]
}

#[inline]
fn mat_fill_zero(x: &mut [f32]) {
    x.fill(0.0);
}

fn lincomb_into(out: &mut [f32], blocks: &[&[f32]], terms: &[(usize, f32)]) {
    mat_fill_zero(out);
    for &(idx, coeff) in terms {
        let src = blocks[idx];
        if coeff == 1.0 {
            for i in 0..out.len() {
                out[i] = (out[i] as f64 + src[i] as f64) as f32;
            }
        } else if coeff == -1.0 {
            for i in 0..out.len() {
                out[i] = (out[i] as f64 - src[i] as f64) as f32;
            }
        } else {
            for i in 0..out.len() {
                out[i] = (out[i] as f64 + (coeff as f64) * (src[i] as f64)) as f32;
            }
        }
    }
}

fn square_mul_into(n: usize, a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n * n);
    debug_assert_eq!(out.len(), n * n);

    // Match the reference naive accumulation order (i-j-k) for closer bitwise behavior.
    out.fill(0.0);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += (a[i * n + k] as f64) * (b[k * n + j] as f64);
            }
            out[i * n + j] = sum as f32;
        }
    }
}

fn copy_block(src: &[f32], src_ld: usize, row0: usize, col0: usize, size: usize, dst: &mut [f32]) {
    debug_assert_eq!(dst.len(), size * size);
    for i in 0..size {
        let src_row = (row0 + i) * src_ld + col0;
        let dst_row = i * size;
        dst[dst_row..dst_row + size].copy_from_slice(&src[src_row..src_row + size]);
    }
}

fn add_block_into(
    dst: &mut [f32],
    dst_ld: usize,
    row0: usize,
    col0: usize,
    size: usize,
    src: &[f32],
) {
    debug_assert_eq!(src.len(), size * size);
    for i in 0..size {
        let dst_row = (row0 + i) * dst_ld + col0;
        let src_row = i * size;
        for j in 0..size {
            dst[dst_row + j] += src[src_row + j];
        }
    }
}

/// Computes C_block += A_block * B_block using an MPTransform when supported.
///
/// - A_block is at (a_row, a_col) in a matrix with leading dimension a_ld.
/// - B_block is at (b_row, b_col) in a matrix with leading dimension b_ld.
/// - C_block is at (c_row, c_col) in a matrix with leading dimension c_ld.
///
/// Returns `true` if MP was applied, `false` if unsupported.
#[allow(clippy::too_many_arguments)]
pub fn mp_block_mul_add(
    a: &[f32],
    a_ld: usize,
    a_row: usize,
    a_col: usize,
    b: &[f32],
    b_ld: usize,
    b_row: usize,
    b_col: usize,
    c: &mut [f32],
    c_ld: usize,
    c_row: usize,
    c_col: usize,
    block_size: usize,
    transform: &MPTransform,
    scratch: &mut MPScratch,
) -> bool {
    if block_size < 2 || !block_size.is_multiple_of(2) {
        return false;
    }
    if transform.mults() != 7 || transform.c_terms.len() != 4 {
        return false;
    }

    let s = scratch.split_mut(block_size);
    let h = s.h;
    let hh = s.hh;

    // 1) Copy A and B quadrants into contiguous buffers.
    // A quads: (0,0),(0,h),(h,0),(h,h) within the block.
    for (q, (dr, dc)) in [(0usize, 0usize), (0, h), (h, 0), (h, h)]
        .into_iter()
        .enumerate()
    {
        let dst = quad_slice_mut(s.a_quads, hh, q);
        copy_block(a, a_ld, a_row + dr, a_col + dc, h, dst);
    }
    for (q, (dr, dc)) in [(0usize, 0usize), (0, h), (h, 0), (h, h)]
        .into_iter()
        .enumerate()
    {
        let dst = quad_slice_mut(s.b_quads, hh, q);
        copy_block(b, b_ld, b_row + dr, b_col + dc, h, dst);
    }

    let a_blocks: [&[f32]; 4] = [
        quad_slice(s.a_quads, hh, 0),
        quad_slice(s.a_quads, hh, 1),
        quad_slice(s.a_quads, hh, 2),
        quad_slice(s.a_quads, hh, 3),
    ];
    let b_blocks: [&[f32]; 4] = [
        quad_slice(s.b_quads, hh, 0),
        quad_slice(s.b_quads, hh, 1),
        quad_slice(s.b_quads, hh, 2),
        quad_slice(s.b_quads, hh, 3),
    ];

    // 2) Compute Mi.
    for mi in 0..7 {
        lincomb_into(s.s, &a_blocks, &transform.a_terms[mi]);
        lincomb_into(s.t, &b_blocks, &transform.b_terms[mi]);
        let m_dst = &mut s.m[mi * hh..(mi + 1) * hh];
        square_mul_into(h, s.s, s.t, m_dst);
    }

    let m_blocks: [&[f32]; 7] = [
        &s.m[..hh],
        &s.m[hh..2 * hh],
        &s.m[2 * hh..3 * hh],
        &s.m[3 * hh..4 * hh],
        &s.m[4 * hh..5 * hh],
        &s.m[5 * hh..6 * hh],
        &s.m[6 * hh..7 * hh],
    ];

    // 3) Recombine into C quadrants and add into destination.
    for cq in 0..4 {
        let out_quad = quad_slice_mut(s.c_quads, hh, cq);
        mat_fill_zero(out_quad);
        for &(mi, coeff) in &transform.c_terms[cq] {
            let src = m_blocks[mi];
            if coeff == 1.0 {
                for i in 0..hh {
                    out_quad[i] = (out_quad[i] as f64 + src[i] as f64) as f32;
                }
            } else if coeff == -1.0 {
                for i in 0..hh {
                    out_quad[i] = (out_quad[i] as f64 - src[i] as f64) as f32;
                }
            } else {
                for i in 0..hh {
                    out_quad[i] = (out_quad[i] as f64 + (coeff as f64) * (src[i] as f64)) as f32;
                }
            }
        }

        let (dr, dc) = match cq {
            0 => (0, 0),
            1 => (0, h),
            2 => (h, 0),
            3 => (h, h),
            _ => unreachable!(),
        };
        add_block_into(c, c_ld, c_row + dr, c_col + dc, h, out_quad);
    }

    true
}
