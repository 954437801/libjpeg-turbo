/*
 * jidctint-lsx.c -- LoongArch LSX implementation of the slow integer
 *                   inverse DCT (jpeg_idct_islow).
 *
 * Numerical semantics follow jidctint.c byte for byte.  The C version relies
 * on MULTIPLY16C16(), i.e. a 16x16 -> 32 product with no pre-shift, and
 * narrows only where it calls DESCALE().  The same structure is used here:
 * every product and every sum stays in 32 bits, and the two narrow points
 * are DESCALE_P1 (after pass 1) and DESCALE_P2 (after pass 2).
 *
 * Layout: one 128-bit register holds eight int16 lanes, i.e. all eight
 * columns of one row.  A single 8-point butterfly therefore covers the whole
 * block, and the two passes are separated by one 8x8 transpose.
 *
 * LSX semantics used here were pinned down with iprobe.c on box185
 * (gcc 8.3):
 *   vmulwev_w_h(a,b) -> { a0*b0, a2*b2, a4*b4, a6*b6 }   (even lanes)
 *   vmulwod_w_h(a,b) -> { a1*b1, a3*b3, a5*b5, a7*b7 }   (odd lanes)
 *   vsrlrni_h_w(hi, lo, n) -> { lo0..lo3, hi0..hi3 } >> n, arithmetic
 *                              rounding, identical to DESCALE(x, n)
 *   vssrarni_b_h is SIGNED saturating, so the range_limit replacement must
 *   bias *before* the unsigned narrow:
 *       vssrarni_bu_h(0, vsadd_h(x, 128), 0) == clamp(x + 128, 0, 255)
 *   vshuf.h is unusable on gcc 8.3 (scrambled lanes), so the even/odd
 *   realignment is done with vshuf.b and explicit byte indices.
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
#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)

#define CENTERJSAMPLE  128

/* jddctmgr.c defines this for the C path; repeat it here because the SIMD
 * kernel is compiled as a separate translation unit. */
#ifndef ISLOW_MULT_TYPE
#define ISLOW_MULT_TYPE  short
#endif

#define FIX_0_298631336  2446
#define FIX_0_390180644  3196
#define FIX_0_541196100  4433
#define FIX_0_765366865  6270
#define FIX_0_899976223  7373
#define FIX_1_175875602  9633
#define FIX_1_501321110  12299
#define FIX_1_847759065  15137
#define FIX_1_961570560  16069
#define FIX_2_053119869  16819
#define FIX_2_562915447  20995
#define FIX_3_072711026  25172

/* [e0,e1,e2,e3,o0,o1,o2,o3] -> [e0,o0,e1,o1,e2,o2,e3,o3] */
static const unsigned char k_shuf_eo[16] = {
  0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
};

/* all ones except lane 0: used to strip the DC coefficient from in[0] */
static const short k_no_dc[8] = { 0, -1, -1, -1, -1, -1, -1, -1 };

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
 * MULTIPLY16C16(): 16x16 -> 32, no scaling. */
static INLINE void
lsx_mul16(__m128i a, __m128i c, __m128i *e, __m128i *o)
{
  *e = __lsx_vmulwev_w_h(a, c);
  *o = __lsx_vmulwod_w_h(a, c);
}
/* ==== END fn=lsx_mul16 ==== */

/* ==== BEGIN fn=lsx_madd16 ==== */
/* e/o += a * c */
static INLINE void
lsx_madd16(__m128i *e, __m128i *o, __m128i a, __m128i c)
{
  *e = __lsx_vmaddwev_w_h(*e, a, c);
  *o = __lsx_vmaddwod_w_h(*o, a, c);
}
/* ==== END fn=lsx_madd16 ==== */

/* ==== BEGIN fn=lsx_transpose8 ==== */
/* 8x8 int16 transpose in three layers, one layer per element width:
 *   layer 1 interleaves 16-bit elements (pairs of rows)
 *   layer 2 interleaves 32-bit elements (quads of rows)
 *   layer 3 interleaves 64-bit elements (all eight rows)
 * Using vilvl_h for every layer (the obvious-looking mistake) leaves the
 * lanes half-transposed and swaps columns in pairs.
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

/* ==== BEGIN fn=lsx_idct_core ==== */
/* One 8-point islow butterfly applied to all eight lanes at once.
 *
 * in0/in4 are passed twice: as int16 (only used where the C code applies
 * MULTIPLY16C16, which truncates to 16 bits anyway) and as 32-bit even/odd
 * pairs (i0e/i0o/i4e/i4o).  The DC term (in0 + in4) << CONST_BITS is NOT
 * truncated by the C code, so it must use the 32-bit view -- this is the
 * only place where a dequantized coefficient can exceed 16 bits.
 * Outputs are the eight 32-bit even/odd result pairs, still scaled by 2^13. */
static INLINE void
lsx_idct_core(__m128i in0, __m128i in1, __m128i in2, __m128i in3,
              __m128i in4, __m128i in5, __m128i in6, __m128i in7,
              __m128i i0e, __m128i i0o, __m128i i4e, __m128i i4o,
              __m128i *oe, __m128i *oo)
{
  __m128i zero = __lsx_vldi(0);
  __m128i c0541 = VB16(FIX_0_541196100);
  __m128i c0847 = VB16(FIX_1_847759065);
  __m128i c0765 = VB16(FIX_0_765366865);
  __m128i c1175 = VB16(FIX_1_175875602);
  __m128i c0298 = VB16(FIX_0_298631336);
  __m128i c2053 = VB16(FIX_2_053119869);
  __m128i c3072 = VB16(FIX_3_072711026);
  __m128i c1501 = VB16(FIX_1_501321110);
  __m128i m0899 = VB16(-FIX_0_899976223);
  __m128i m2562 = VB16(-FIX_2_562915447);
  __m128i m1961 = VB16(-FIX_1_961570560);
  __m128i m0390 = VB16(-FIX_0_390180644);

  /* 32-bit sign-extended views of the DC pair are now supplied by the
   * caller (i0e/i0o/i4e/i4o), because DEQUANTIZE() does not truncate. */

  __m128i z1e, z1o, t2e, t2o, t3e, t3o, t0e, t0o, t1e, t1o;
  __m128i t10e, t10o, t11e, t11o, t12e, t12o, t13e, t13o;
  __m128i z2e, z2o, z3e, z3o, z4e, z4o, z5e, z5o;
  __m128i f0e, f0o, f1e, f1o, f2e, f2o, f3e, f3o;

  /* ---- even part -------------------------------------------------- */
  /* z1 = (in2 + in6) * FIX_0_541196100, distributed to stay in 16 bits */
  lsx_mul16(in2, c0541, &z1e, &z1o);
  lsx_madd16(&z1e, &z1o, in6, c0541);

  /* tmp2 = z1 - in6 * FIX_1_847759065 */
  lsx_mul16(in6, c0847, &t2e, &t2o);
  t2e = __lsx_vsub_w(z1e, t2e);
  t2o = __lsx_vsub_w(z1o, t2o);
  /* tmp3 = z1 + in2 * FIX_0_765366865 */
  lsx_mul16(in2, c0765, &t3e, &t3o);
  t3e = __lsx_vadd_w(z1e, t3e);
  t3o = __lsx_vadd_w(z1o, t3o);

  /* tmp0 = (in0 + in4) << CONST_BITS ; tmp1 = (in0 - in4) << CONST_BITS */
  t0e = __lsx_vslli_w(__lsx_vadd_w(i0e, i4e), CONST_BITS);
  t0o = __lsx_vslli_w(__lsx_vadd_w(i0o, i4o), CONST_BITS);
  t1e = __lsx_vslli_w(__lsx_vsub_w(i0e, i4e), CONST_BITS);
  t1o = __lsx_vslli_w(__lsx_vsub_w(i0o, i4o), CONST_BITS);

  t10e = __lsx_vadd_w(t0e, t3e);
  t10o = __lsx_vadd_w(t0o, t3o);
  t13e = __lsx_vsub_w(t0e, t3e);
  t13o = __lsx_vsub_w(t0o, t3o);
  t11e = __lsx_vadd_w(t1e, t2e);
  t11o = __lsx_vadd_w(t1o, t2o);
  t12e = __lsx_vsub_w(t1e, t2e);
  t12o = __lsx_vsub_w(t1o, t2o);

  /* ---- odd part ---------------------------------------------------- */
  /* The C code forms z1..z4 as int sums of two dequantized coefficients.
   * With the recommended 8-bit scaling every intermediate stays within
   * 16 bits (see the comment above MULTIPLY16C16 in jidctint.c), so the
   * sums are formed in 16 bits exactly like the NEON port does. */
  {
    __m128i z1s = __lsx_vadd_h(in7, in1);
    __m128i z2s = __lsx_vadd_h(in5, in3);
    __m128i z3s = __lsx_vadd_h(in7, in3);
    __m128i z4s = __lsx_vadd_h(in5, in1);
    __m128i z5s = __lsx_vadd_h(z3s, z4s);

    /* z5 = (z3 + z4) * FIX_1_175875602 */
    lsx_mul16(z5s, c1175, &z5e, &z5o);

    /* z1..z4 *= their constants, z3/z4 then accumulate z5 */
    lsx_mul16(z1s, m0899, &z1e, &z1o);
    lsx_mul16(z2s, m2562, &z2e, &z2o);
    lsx_mul16(z3s, m1961, &z3e, &z3o);
    lsx_mul16(z4s, m0390, &z4e, &z4o);
    z3e = __lsx_vadd_w(z3e, z5e);
    z3o = __lsx_vadd_w(z3o, z5o);
    z4e = __lsx_vadd_w(z4e, z5e);
    z4o = __lsx_vadd_w(z4o, z5o);

    /* tmp0..tmp3 = in7/in5/in3/in1 * their constants, then + z1/z3, z2/z4 */
    lsx_mul16(in7, c0298, &f0e, &f0o);
    f0e = __lsx_vadd_w(f0e, __lsx_vadd_w(z1e, z3e));
    f0o = __lsx_vadd_w(f0o, __lsx_vadd_w(z1o, z3o));

    lsx_mul16(in5, c2053, &f1e, &f1o);
    f1e = __lsx_vadd_w(f1e, __lsx_vadd_w(z2e, z4e));
    f1o = __lsx_vadd_w(f1o, __lsx_vadd_w(z2o, z4o));

    lsx_mul16(in3, c3072, &f2e, &f2o);
    f2e = __lsx_vadd_w(f2e, __lsx_vadd_w(z2e, z3e));
    f2o = __lsx_vadd_w(f2o, __lsx_vadd_w(z2o, z3o));

    lsx_mul16(in1, c1501, &f3e, &f3o);
    f3e = __lsx_vadd_w(f3e, __lsx_vadd_w(z1e, z4e));
    f3o = __lsx_vadd_w(f3o, __lsx_vadd_w(z1o, z4o));
  }

  oe[0] = __lsx_vadd_w(t10e, f3e);  oo[0] = __lsx_vadd_w(t10o, f3o);
  oe[7] = __lsx_vsub_w(t10e, f3e);  oo[7] = __lsx_vsub_w(t10o, f3o);
  oe[1] = __lsx_vadd_w(t11e, f2e);  oo[1] = __lsx_vadd_w(t11o, f2o);
  oe[6] = __lsx_vsub_w(t11e, f2e);  oo[6] = __lsx_vsub_w(t11o, f2o);
  oe[2] = __lsx_vadd_w(t12e, f1e);  oo[2] = __lsx_vadd_w(t12o, f1o);
  oe[5] = __lsx_vsub_w(t12e, f1e);  oo[5] = __lsx_vsub_w(t12o, f1o);
  oe[3] = __lsx_vadd_w(t13e, f0e);  oo[3] = __lsx_vadd_w(t13o, f0o);
  oe[4] = __lsx_vsub_w(t13e, f0e);  oo[4] = __lsx_vsub_w(t13o, f0o);
}
/* ==== END fn=lsx_idct_core ==== */

/* ==== BEGIN fn=jsimd_idct_islow_lsx ==== */
GLOBAL(void)
jsimd_idct_islow_lsx(void *dct_table, JCOEFPTR coef_block,
                     JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quantptr = (ISLOW_MULT_TYPE *) dct_table;
  __m128i in[8], oe[8], oo[8], out[8];
  __m128i c0 = __lsx_vld(coef_block, 0);
  __m128i q0 = __lsx_vld(quantptr, 0);
  __m128i c4 = __lsx_vld(coef_block + 4 * DCTSIZE, 0);
  __m128i q4 = __lsx_vld(quantptr + 4 * DCTSIZE, 0);
  __m128i zero = __lsx_vldi(0);
  /* Rows 0 and 4 feed the DC term, which the C code never truncates, so
   * they are dequantized in 32 bits as well. */
  __m128i i0e = __lsx_vmulwev_w_h(c0, q0), i0o = __lsx_vmulwod_w_h(c0, q0);
  __m128i i4e = __lsx_vmulwev_w_h(c4, q4), i4o = __lsx_vmulwod_w_h(c4, q4);
  __m128i bias = __lsx_vreplgr2vr_h(CENTERJSAMPLE);
  __m128i zero8 = __lsx_vreplgr2vr_b(0);
  int i;

  /* 16-bit view, identical to MULTIPLY16C16()'s implicit (INT16) cast */
  for (i = 0; i < 8; i++) {
    __m128i c = __lsx_vld(coef_block + i * DCTSIZE, 0);
    __m128i q = __lsx_vld(quantptr + i * DCTSIZE, 0);
    in[i] = __lsx_vmul_h(c, q);
  }

  /* Fast path: the DC coefficient is the ONLY non-zero one.  Testing
   * rows 1-7 alone is not enough -- the C version decides per column, so
   * an AC value in row 0 (lanes 1..7 of in[0]) still needs the butterfly
   * for those columns. */
  {
    __m128i ac = __lsx_vand_v(in[0], __lsx_vld(k_no_dc, 0));
    ac = __lsx_vor_v(ac, __lsx_vor_v(in[1], in[2]));
    ac = __lsx_vor_v(ac, __lsx_vor_v(in[3], in[4]));
    ac = __lsx_vor_v(ac, __lsx_vor_v(in[5], in[6]));
    ac = __lsx_vor_v(ac, in[7]);
    if ((__lsx_vpickve2gr_d(ac, 0) | __lsx_vpickve2gr_d(ac, 1)) == 0) {
      /* Scalars on purpose: only lane 0 carries the DC value, and the whole
       * 8x8 output is that one pixel.  Keeping it scalar also sidesteps
       * int16 narrowing of an up-to-260k dequantized DC. */
      int dc = __lsx_vpickve2gr_w(i0e, 0);
      /* range_limit[] adds CENTERJSAMPLE while clamping: value is
       * clamp(DESCALE(dc << PASS1_BITS, PASS1_BITS + 3) + 128, 0, 255). */
      int pix = (int) (((long long) dc * (1 << PASS1_BITS) +
                        (1 << (PASS1_BITS + 2))) >> (PASS1_BITS + 3));
      __m128i b;

      pix += CENTERJSAMPLE;
      if (pix < 0)
        pix = 0;
      else if (pix > MAXJSAMPLE)
        pix = MAXJSAMPLE;
      b = __lsx_vreplgr2vr_b((char) (unsigned char) pix);
      for (i = 0; i < 8; i++)
        __lsx_vstelm_d(b, output_buf[i] + output_col, 0, 0);
      return;
    }
  }

  /* ---- pass 1 ------------------------------------------------------ */
  lsx_idct_core(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7],
                i0e, i0o, i4e, i4o, oe, oo);
  for (i = 0; i < 8; i++)
    out[i] = PACK_ROUND(oe[i], oo[i], DESCALE_P1);

  /* Pass 1 holds "lane j = column j" per output row.  Pass 2 transforms
   * along the columns, so one transpose puts the rows in the lane index. */
  lsx_transpose8(&out[0], &out[1], &out[2], &out[3],
                 &out[4], &out[5], &out[6], &out[7]);

  /* ---- pass 2 ------------------------------------------------------ */
  /* DESCALE_P2 is 18, which is more than a narrow can shift, so the shift
   * has to happen in the 32-bit domain: add 2^17, arithmetic-shift by 18,
   * and only then narrow (shift 0).  Doing it as "shift 2 then narrow"
   * would leave values far outside int16 and truncate them. */
  lsx_idct_core(out[0], out[1], out[2], out[3], out[4], out[5], out[6],
                out[7],
                __lsx_vaddwev_w_h(zero, out[0]),
                __lsx_vaddwod_w_h(zero, out[0]),
                __lsx_vaddwev_w_h(zero, out[4]),
                __lsx_vaddwod_w_h(zero, out[4]),
                oe, oo);
  {
    __m128i half = __lsx_vreplgr2vr_w(1 << (DESCALE_P2 - 1));
    for (i = 0; i < 8; i++) {
      __m128i ye = __lsx_vsrai_w(__lsx_vadd_w(oe[i], half), DESCALE_P2);
      __m128i yo = __lsx_vsrai_w(__lsx_vadd_w(oo[i], half), DESCALE_P2);
      out[i] = PACK_ROUND(ye, yo, 0);
    }
  }

  /* Transpose back so that each register holds one output row. */
  lsx_transpose8(&out[0], &out[1], &out[2], &out[3],
                 &out[4], &out[5], &out[6], &out[7]);

  for (i = 0; i < 8; i++) {
    __m128i v = __lsx_vsadd_h(out[i], bias);
    __m128i b = __lsx_vssrarni_bu_h(zero8, v, 0);
    /* only the low eight bytes are valid: never write 16 */
    __lsx_vstelm_d(b, output_buf[i] + output_col, 0, 0);
  }
}
/* ==== END fn=jsimd_idct_islow_lsx ==== */