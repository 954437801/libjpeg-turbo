/*
 * jfdctint-lsx.c -- LoongArch LSX implementation of the slow integer
 *                   forward DCT (jpeg_fdct_islow).
 *
 * Numerical semantics follow jfdctint.c exactly.  The C version multiplies
 * through MULTIPLY16C16(), which in this build resolves to a plain 32-bit
 * product (neither SHORTxSHORT_32 nor SHORTxLCONST_32 is configured), so the
 * ports below keep every product and every sum in 32 bits and narrow only
 * where the C code calls DESCALE() / LEFT_SHIFT().
 *
 * Layout: one 128-bit register holds eight int16 lanes.
 *   pass 1 transforms along each row, so the eight rows are transposed
 *          first; afterwards lane = row and vector index = column.
 *   pass 2 transforms along each column, which is the natural layout of
 *          the pass-1 results after one more transpose.
 * Every product operand is a sum of at most four dequantized samples, hence
 * provably within int16 (|pass-1 output| <= 4096, see the scaling comment in
 * jfdctint.c), except out0/out4 which the C code never truncates -- those are
 * carried as 32-bit even/odd lane pairs.
 *
 * LSX semantics used here were pinned down with iprobe.c on box185
 * (gcc 8.3); the ones added for this kernel were verified with sprobe.c:
 *   vmulwev_w_h(a,b) -> { a0*b0, a2*b2, a4*b4, a6*b6 }   (even lanes)
 *   vmulwod_w_h(a,b) -> { a1*b1, a3*b3, a5*b5, a7*b7 }   (odd lanes)
 *   vsrlrni_h_w(hi, lo, n) -> DESCALE(x, n) with lo in the low half
 *   vslli_w(hi, lo, n) is a plain 32-bit left shift
 *
 * Copyright (C) 2026, Loongson Technology Corporation Limited.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"
#include "../../jsimddct.h"
#include "lsxint.h"

#define CONST_BITS  13
#define PASS1_BITS  2
#define DESCALE_P1  (CONST_BITS - PASS1_BITS)   /* 11 */
#define DESCALE_P2  (CONST_BITS + PASS1_BITS)   /* 15 */

#define F_0_298  2446                   /* FIX(0.298631336) */
#define F_0_390  3196                   /* FIX(0.390180644) */
#define F_0_541  4433                   /* FIX(0.541196100) */
#define F_0_765  6270                   /* FIX(0.765366865) */
#define F_0_899  7373                   /* FIX(0.899976223) */
#define F_1_175  9633                   /* FIX(1.175875602) */
#define F_1_501  12299                  /* FIX(1.501321110) */
#define F_1_847  15137                  /* FIX(1.847759065) */
#define F_1_961  16069                  /* FIX(1.961570560) */
#define F_2_053  16819                  /* FIX(2.053119869) */
#define F_2_562  20995                  /* FIX(2.562915447) */
#define F_3_072  25172                  /* FIX(3.072711026) */

/* [e0,e1,e2,e3,o0,o1,o2,o3] -> [e0,o0,e1,o1,e2,o2,e3,o3] */
static const unsigned char k_shuf_eo[16] = {
  0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
};

/* Broadcast a 16-bit constant (the C code's FIX_* values). */
#define VB16(c)  __lsx_vreplgr2vr_h((short) (c))

/* Narrow a 32-bit even/odd pair to eight int16 in natural column order.
 * "n" must be a literal: vsrlrni takes an immediate. */
#define PACK_ROUND(e, o, n) __extension__ ({                         \
  __m128i _t = __lsx_vsrlrni_h_w((o), (e), (n));                    \
  __lsx_vshuf_b(_t, _t, __lsx_vld(k_shuf_eo, 0));                    \
})

/* ==== BEGIN fn=lsx_mul16 ==== */
/* e/o = a * c, with a = eight int16 and c broadcast.  Mirrors
 * MULTIPLY16C16() for operands that provably fit in 16 bits. */
static INLINE void
lsx_mul16(__m128i a, __m128i c, __m128i *e, __m128i *o)
{
  *e = __lsx_vmulwev_w_h(a, c);
  *o = __lsx_vmulwod_w_h(a, c);
}
/* ==== END fn=lsx_mul16 ==== */

/* ==== BEGIN fn=lsx_transpose8 ==== */
/* 8x8 int16 transpose in three layers, one layer per element width:
 *   layer 1 interleaves 16-bit elements (pairs of rows)
 *   layer 2 interleaves 32-bit elements (quads of rows)
 *   layer 3 interleaves 64-bit elements (all eight rows)
 * vilvl_*(x, y) puts y in the even lanes, so the "low" operand is second. */
static INLINE void
lsx_transpose8(__m128i *r0, __m128i *r1, __m128i *r2, __m128i *r3,
               __m128i *r4, __m128i *r5, __m128i *r6, __m128i *r7)
{
  __m128i a0 = __lsx_vilvl_h(*r1, *r0), a1 = __lsx_vilvh_h(*r1, *r0);
  __m128i a2 = __lsx_vilvl_h(*r3, *r2), a3 = __lsx_vilvh_h(*r3, *r2);
  __m128i a4 = __lsx_vilvl_h(*r5, *r4), a5 = __lsx_vilvh_h(*r5, *r4);
  __m128i a6 = __lsx_vilvl_h(*r7, *r6), a7 = __lsx_vilvh_h(*r7, *r6);

  __m128i b0 = __lsx_vilvl_w(a2, a0), b1 = __lsx_vilvh_w(a2, a0);
  __m128i b2 = __lsx_vilvl_w(a3, a1), b3 = __lsx_vilvh_w(a3, a1);
  __m128i b4 = __lsx_vilvl_w(a6, a4), b5 = __lsx_vilvh_w(a6, a4);
  __m128i b6 = __lsx_vilvl_w(a7, a5), b7 = __lsx_vilvh_w(a7, a5);

  *r0 = __lsx_vilvl_d(b4, b0);
  *r1 = __lsx_vilvh_d(b4, b0);
  *r2 = __lsx_vilvl_d(b5, b1);
  *r3 = __lsx_vilvh_d(b5, b1);
  *r4 = __lsx_vilvl_d(b6, b2);
  *r5 = __lsx_vilvh_d(b6, b2);
  *r6 = __lsx_vilvl_d(b7, b3);
  *r7 = __lsx_vilvh_d(b7, b3);
}
/* ==== END fn=lsx_transpose8 ==== */

/* ==== BEGIN fn=lsx_fdct_core ==== */
/* One 8-point islow butterfly applied to all eight lanes at once.
 * Returns the eight results still scaled by 2^CONST_BITS (plus the sqrt(8)
 * factor of the two-pass arrangement); the caller applies DESCALE.
 *
 * The stage-1 and stage-2 adds are done in 16 bits on purpose: with the
 * recommended 8-bit scaling every intermediate is bounded by 2^14, and the
 * four sums that feed a MULTIPLY are wider than that, so they are formed as
 * 32-bit lane pairs and multiplied one term at a time
 * ((a + b) * c == a * c + b * c exactly, and each product fits in 32 bits). */
static INLINE void
lsx_fdct_core(__m128i in0, __m128i in1, __m128i in2, __m128i in3,
              __m128i in4, __m128i in5, __m128i in6, __m128i in7,
              __m128i *oe, __m128i *oo)
{
  const __m128i c0541 = VB16(F_0_541);
  const __m128i c0765 = VB16(F_0_765);
  const __m128i m1847 = VB16(-F_1_847);
  const __m128i c1175 = VB16(F_1_175);
  const __m128i c0298 = VB16(F_0_298);
  const __m128i c2053 = VB16(F_2_053);
  const __m128i c3072 = VB16(F_3_072);
  const __m128i c1501 = VB16(F_1_501);
  const __m128i m0899 = VB16(-F_0_899);
  const __m128i m2562 = VB16(-F_2_562);
  const __m128i m1961 = VB16(-F_1_961);
  const __m128i m0390 = VB16(-F_0_390);

  __m128i v0, v1, v2, v3, v4, v5, v6, v7;
  __m128i t10, t11, t12, t13;
  __m128i z1, z2, z3, z4, z5e, z5o;
  __m128i p1e, p1o, p2e, p2o, p3e, p3o, p4e, p4o;
  __m128i m4e, m4o, m5e, m5o, m6e, m6o, m7e, m7o;
  __m128i z1e, z1o, ue, uo;

  /* ---- stage 1: tmp0/tmp7 ... tmp3/tmp4 ---------------------------- */
  v0 = __lsx_vadd_h(in0, in7);  v7 = __lsx_vsub_h(in0, in7);
  v1 = __lsx_vadd_h(in1, in6);  v6 = __lsx_vsub_h(in1, in6);
  v2 = __lsx_vadd_h(in2, in5);  v5 = __lsx_vsub_h(in2, in5);
  v3 = __lsx_vadd_h(in3, in4);  v4 = __lsx_vsub_h(in3, in4);

  /* ---- stage 2: even part ------------------------------------------ */
  t10 = __lsx_vadd_h(v0, v3);  t13 = __lsx_vsub_h(v0, v3);
  t11 = __lsx_vadd_h(v1, v2);  t12 = __lsx_vsub_h(v1, v2);

  /* data[0]/data[4] keep 32-bit precision: the C code shifts them without
   * any DESCALE truncation. */
  {
    __m128i a = __lsx_vaddwev_w_h(__lsx_vldi(0), t10);
    __m128i b = __lsx_vaddwod_w_h(__lsx_vldi(0), t10);
    __m128i c = __lsx_vaddwev_w_h(__lsx_vldi(0), t11);
    __m128i d = __lsx_vaddwod_w_h(__lsx_vldi(0), t11);

    oe[0] = __lsx_vadd_w(a, c);  oo[0] = __lsx_vadd_w(b, d);
    oe[4] = __lsx_vsub_w(a, c);  oo[4] = __lsx_vsub_w(b, d);
  }

  /* Even rotators: z1 = (tmp12 + tmp13) * c4, then
   * data[2] = z1 + tmp13 * c2, data[6] = z1 + tmp12 * (-c6). */
  lsx_mul16(t12, c0541, &z1e, &z1o);
  lsx_mul16(t13, c0541, &ue, &uo);
  z1e = __lsx_vadd_w(z1e, ue);
  z1o = __lsx_vadd_w(z1o, uo);

  lsx_mul16(t13, c0765, &ue, &uo);
  oe[2] = __lsx_vadd_w(z1e, ue);  oo[2] = __lsx_vadd_w(z1o, uo);

  lsx_mul16(t12, m1847, &ue, &uo);
  oe[6] = __lsx_vadd_w(z1e, ue);  oo[6] = __lsx_vadd_w(z1o, uo);

  /* ---- odd part ---------------------------------------------------- */
  z1 = __lsx_vadd_h(v4, v7);
  z2 = __lsx_vadd_h(v5, v6);
  z3 = __lsx_vadd_h(v4, v6);
  z4 = __lsx_vadd_h(v5, v7);

  /* z5 = (z3 + z4) * c3, split into two products for the same reason as z1 */
  lsx_mul16(z3, c1175, &z5e, &z5o);
  lsx_mul16(z4, c1175, &ue, &uo);
  z5e = __lsx_vadd_w(z5e, ue);
  z5o = __lsx_vadd_w(z5o, uo);

  lsx_mul16(z1, m0899, &p1e, &p1o);
  lsx_mul16(z2, m2562, &p2e, &p2o);
  lsx_mul16(z3, m1961, &p3e, &p3o);
  lsx_mul16(z4, m0390, &p4e, &p4o);
  p3e = __lsx_vadd_w(p3e, z5e);  p3o = __lsx_vadd_w(p3o, z5o);
  p4e = __lsx_vadd_w(p4e, z5e);  p4o = __lsx_vadd_w(p4o, z5o);

  lsx_mul16(v4, c0298, &m4e, &m4o);
  lsx_mul16(v5, c2053, &m5e, &m5o);
  lsx_mul16(v6, c3072, &m6e, &m6o);
  lsx_mul16(v7, c1501, &m7e, &m7o);

  oe[7] = __lsx_vadd_w(__lsx_vadd_w(m4e, p1e), p3e);
  oo[7] = __lsx_vadd_w(__lsx_vadd_w(m4o, p1o), p3o);
  oe[5] = __lsx_vadd_w(__lsx_vadd_w(m5e, p2e), p4e);
  oo[5] = __lsx_vadd_w(__lsx_vadd_w(m5o, p2o), p4o);
  oe[3] = __lsx_vadd_w(__lsx_vadd_w(m6e, p2e), p3e);
  oo[3] = __lsx_vadd_w(__lsx_vadd_w(m6o, p2o), p3o);
  oe[1] = __lsx_vadd_w(__lsx_vadd_w(m7e, p1e), p4e);
  oo[1] = __lsx_vadd_w(__lsx_vadd_w(m7o, p1o), p4o);
}
/* ==== END fn=lsx_fdct_core ==== */

/* ==== BEGIN fn=jsimd_fdct_islow_lsx ==== */
GLOBAL(void)
jsimd_fdct_islow_lsx(DCTELEM *data)
{
  __m128i v[8], oe[8], oo[8];
  int i;

  for (i = 0; i < 8; i++)
    v[i] = __lsx_vld(data + i * DCTSIZE, 0);

  /* ---- pass 1 ------------------------------------------------------ */
  /* The butterfly runs along each row, so the block is transposed first:
   * afterwards v[c] holds column c of the block and the lane index is the
   * row.  Results: v[k] lane r = coefficient k of row r. */
  lsx_transpose8(&v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
  lsx_fdct_core(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], oe, oo);

  for (i = 0; i < 8; i++) {
    if (i == 0 || i == 4)
      /* data[0]/data[4] use LEFT_SHIFT, i.e. no rounding at all */
      v[i] = PACK_ROUND(__lsx_vslli_w(oe[i], PASS1_BITS),
                        __lsx_vslli_w(oo[i], PASS1_BITS), 0);
    else
      v[i] = PACK_ROUND(oe[i], oo[i], DESCALE_P1);
  }

  /* ---- pass 2 ------------------------------------------------------ */
  /* Pass 2 runs along the columns, which after one more transpose is again
   * the vector index.  The results come out already in row order. */
  lsx_transpose8(&v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
  lsx_fdct_core(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], oe, oo);

  for (i = 0; i < 8; i++) {
    if (i == 0 || i == 4)
      v[i] = PACK_ROUND(oe[i], oo[i], PASS1_BITS);
    else
      v[i] = PACK_ROUND(oe[i], oo[i], DESCALE_P2);
  }

  for (i = 0; i < 8; i++)
    __lsx_vst(v[i], data + i * DCTSIZE, 0);
}
/* ==== END fn=jsimd_fdct_islow_lsx ==== */
