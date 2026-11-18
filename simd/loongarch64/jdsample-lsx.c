/*
 * jdsample-lsx.c
 *
 * Upsampling routines, LoongArch LSX (128-bit) SIMD.
 *
 * Five kernels are provided, matching the ones the other architectures
 * implement: h2v2/h2v1 fancy (linear interpolation with a 3:1 blend) and
 * h2v2/h2v1 "box filter" replication, plus h1v2 fancy (vertical-only blend).
 *
 * Numerical output is bit for bit identical to the C implementations in
 * jdsample.c.  The odd/even output positions use different biases in the C
 * code (+7 truncating vs. +8 rounding for h2v2 fancy, +1 vs. +2 for the
 * 2:1 kernels), so the two halves are narrowed with vsrlni (truncating) and
 * vsrlrni (rounding) respectively rather than with one shared rounding step.
 *
 * Unlike the NEON port, the vector loops here stop one block short of the row
 * end and finish with a scalar tail, so no store ever runs past
 * output_width and no load ever runs past downsampled_width.
 *
 * Copyright (C) 2026, Loongson Technology Corporation Limited.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"          /* DCTELEM, needed by simd/jsimd.h */
#include "../jsimd.h"
#include "lsxint.h"

#define LSX_DUP3(v, three)  __lsx_vmul_h((v), (three))
#define LSX_WID_LO(v)       __lsx_vsllwil_hu_bu((v), 0)
#define LSX_WID_HI(v)       __lsx_vexth_hu_bu((v))


/* ==== BEGIN fn=lsx_ilv_store ==== */
/* Store the 32 output bytes of a 16-sample fancy block.  Every narrowing
 * result carries its 8 meaningful values in the low half (bytes 0..7); the
 * upper half just duplicates the source and is never read, because vilvl
 * only takes the low 8 bytes of each operand.  The second operand's element
 * comes first in the interleave, hence "b" before "a". */
static void
lsx_ilv_store(__m128i a_lo, __m128i b_lo, __m128i a_hi, __m128i b_hi,
              JSAMPLE *dst)
{
  __lsx_vst(__lsx_vilvl_b(b_lo, a_lo), dst, 0);
  __lsx_vst(__lsx_vilvl_b(b_hi, a_hi), dst + 16, 0);
}
/* ==== END fn=lsx_ilv_store ==== */


/* ==== BEGIN fn=lsx_dup_store ==== */
/* Store 32 bytes produced by duplicating each of 16 samples twice. */
static void
lsx_dup_store(__m128i s, JSAMPLE *dst)
{
  __lsx_vst(__lsx_vilvl_b(s, s), dst, 0);
  __lsx_vst(__lsx_vilvh_b(s, s), dst + 16, 0);
}
/* ==== END fn=lsx_dup_store ==== */


/* ==== BEGIN fn=lsx_h2v1_fancy_block ==== */
/* One block of 16 samples -> 32 output pixels, for 2h1v fancy upsampling:
 *   out[2k+1] = (3 * in[k] + in[k+1] + 2) >> 2   (rounding)
 *   out[2k+2] = (3 * in[k+1] + in[k] + 1) >> 2   (truncating)
 * in[k .. k+16] is read, out[2k+1 .. 2k+32] is written. */
static void
lsx_h2v1_fancy_block(const JSAMPLE *inp, JSAMPLE *outp)
{
  const __m128i three = __lsx_vreplgr2vr_h(3);
  const __m128i one = __lsx_vreplgr2vr_h(1);
  __m128i s0 = __lsx_vld(inp, 0);
  __m128i s1 = __lsx_vld(inp + 1, 0);
  __m128i a0, a1, odd, even, o_lo, e_lo, o_hi, e_hi;

  /* samples k..k+7 */
  a0 = LSX_WID_LO(s0);
  a1 = LSX_WID_LO(s1);
  odd = __lsx_vadd_h(LSX_DUP3(a0, three), a1);
  even = __lsx_vadd_h(__lsx_vadd_h(LSX_DUP3(a1, three), a0), one);
  o_lo = __lsx_vsrlrni_b_h(odd, odd, 2);    /* odd carries the +2 bias */
  e_lo = __lsx_vsrlni_b_h(even, even, 2);   /* even carries the +1 bias */

  /* samples k+8..k+15 */
  a0 = LSX_WID_HI(s0);
  a1 = LSX_WID_HI(s1);
  odd = __lsx_vadd_h(LSX_DUP3(a0, three), a1);
  even = __lsx_vadd_h(__lsx_vadd_h(LSX_DUP3(a1, three), a0), one);
  o_hi = __lsx_vsrlrni_b_h(odd, odd, 2);
  e_hi = __lsx_vsrlni_b_h(even, even, 2);

  lsx_ilv_store(o_lo, e_lo, o_hi, e_hi, outp);
}
/* ==== END fn=lsx_h2v1_fancy_block ==== */


/* ==== BEGIN fn=lsx_h2v2_fancy_block ==== */
/* One block of 16 samples -> 32 output pixels, for 2h2v fancy upsampling.
 *   colsum[k] = 3 * center[k] + neighbour[k]
 *   out[2k+1] = (3 * colsum[k] + colsum[k+1] + 7) >> 4   (truncating)
 *   out[2k+2] = (3 * colsum[k+1] + colsum[k] + 8) >> 4   (rounding) */
static void
lsx_h2v2_fancy_block(const JSAMPLE *nbrp, const JSAMPLE *cenp, JSAMPLE *outp)
{
  const __m128i three = __lsx_vreplgr2vr_h(3);
  const __m128i seven = __lsx_vreplgr2vr_h(7);
  __m128i c0 = __lsx_vld(cenp, 0);
  __m128i c1 = __lsx_vld(cenp + 1, 0);
  __m128i n0 = __lsx_vld(nbrp, 0);
  __m128i n1 = __lsx_vld(nbrp + 1, 0);
  __m128i cs0, cs1, odd, even, o_lo, e_lo, o_hi, e_hi;

  /* samples k..k+7 */
  cs0 = __lsx_vadd_h(LSX_DUP3(LSX_WID_LO(c0), three), LSX_WID_LO(n0));
  cs1 = __lsx_vadd_h(LSX_DUP3(LSX_WID_LO(c1), three), LSX_WID_LO(n1));
  odd = __lsx_vadd_h(__lsx_vadd_h(LSX_DUP3(cs0, three), cs1), seven);
  even = __lsx_vadd_h(LSX_DUP3(cs1, three), cs0);
  o_lo = __lsx_vsrlni_b_h(odd, odd, 4);     /* odd carries the +7 bias */
  e_lo = __lsx_vsrlrni_b_h(even, even, 4);  /* even carries the +8 bias */

  /* samples k+8..k+15 */
  cs0 = __lsx_vadd_h(LSX_DUP3(LSX_WID_HI(c0), three), LSX_WID_HI(n0));
  cs1 = __lsx_vadd_h(LSX_DUP3(LSX_WID_HI(c1), three), LSX_WID_HI(n1));
  odd = __lsx_vadd_h(__lsx_vadd_h(LSX_DUP3(cs0, three), cs1), seven);
  even = __lsx_vadd_h(LSX_DUP3(cs1, three), cs0);
  o_hi = __lsx_vsrlni_b_h(odd, odd, 4);
  e_hi = __lsx_vsrlrni_b_h(even, even, 4);

  lsx_ilv_store(o_lo, e_lo, o_hi, e_hi, outp);
}
/* ==== END fn=lsx_h2v2_fancy_block ==== */


/* ==== BEGIN fn=lsx_h1v2_fancy_block ==== */
/* One block of 16 columns for 1h2v fancy upsampling:
 *   out[k] = (3 * center[k] + neighbour[k] + bias) >> 2
 * bias is 1 for the upper output row (truncating) and 2 for the lower one
 * (equivalent to a rounding shift). */
static void
lsx_h1v2_fancy_block(const JSAMPLE *cenp, const JSAMPLE *nbrp, JSAMPLE *outp,
                     int bias)
{
  const __m128i three = __lsx_vreplgr2vr_h(3);
  const __m128i one = __lsx_vreplgr2vr_h(1);
  __m128i c = __lsx_vld(cenp, 0);
  __m128i n = __lsx_vld(nbrp, 0);
  __m128i s_lo = __lsx_vadd_h(LSX_DUP3(LSX_WID_LO(c), three), LSX_WID_LO(n));
  __m128i s_hi = __lsx_vadd_h(LSX_DUP3(LSX_WID_HI(c), three), LSX_WID_HI(n));

  if (bias == 1) {
    /* (x + 1) >> 2: bias added explicitly, then a truncating narrow */
    s_lo = __lsx_vadd_h(s_lo, one);
    s_hi = __lsx_vadd_h(s_hi, one);
    __lsx_vst(__lsx_vsrlni_b_h(s_hi, s_lo, 2), outp, 0);
  } else {
    /* (x + 2) >> 2: the rounding narrow supplies the bias */
    __lsx_vst(__lsx_vsrlrni_b_h(s_hi, s_lo, 2), outp, 0);
  }
}
/* ==== END fn=lsx_h1v2_fancy_block ==== */


/* ==== BEGIN fn=jsimd_h2v1_fancy_upsample_lsx ==== */
GLOBAL(void)
jsimd_h2v1_fancy_upsample_lsx(int max_v_samp_factor,
                              JDIMENSION downsampled_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JDIMENSION w = downsampled_width;
  JDIMENSION i, i0, tail;
  int inrow;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    const JSAMPLE *inptr = input_data[inrow];
    JSAMPLE *outptr = output_data[inrow];

    /* The first pixel has no left neighbour.  The second one still uses the
     * 3:1 blend with the +2 (rounding) bias, so it must be emitted here:
     * the vector loop below starts at outptr[2] when the row is too short
     * for a full 16-sample block. */
    outptr[0] = inptr[0];
    outptr[1] =
      (JSAMPLE)((GETJSAMPLE(inptr[0]) * 3 + GETJSAMPLE(inptr[1]) + 2) >> 2);

    /* Vector blocks need in[i0 + 16] to exist for the last sample. */
    tail = 1;
    for (i0 = 0; i0 + 16 < w; i0 += 16) {
      lsx_h2v1_fancy_block(inptr + i0, outptr + 2 * i0 + 1);
      tail = i0 + 16;
    }

    /* Scalar tail: even positions 2i and odd positions 2i+1. */
    for (i = tail; i + 1 < w; i++) {
      int invalue = GETJSAMPLE(inptr[i]) * 3;
      outptr[2 * i] = (JSAMPLE)((invalue + GETJSAMPLE(inptr[i - 1]) + 1) >> 2);
      outptr[2 * i + 1] = (JSAMPLE)((invalue + GETJSAMPLE(inptr[i + 1]) + 2) >> 2);
    }

    /* Last two pixels: no right neighbour, then a plain copy. */
    outptr[2 * w - 2] =
      (JSAMPLE)((GETJSAMPLE(inptr[w - 1]) * 3 + GETJSAMPLE(inptr[w - 2]) + 1) >> 2);
    outptr[2 * w - 1] = inptr[w - 1];
  }
}
/* ==== END fn=jsimd_h2v1_fancy_upsample_lsx ==== */


/* ==== BEGIN fn=jsimd_h2v2_fancy_upsample_lsx ==== */
GLOBAL(void)
jsimd_h2v2_fancy_upsample_lsx(int max_v_samp_factor,
                              JDIMENSION downsampled_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JDIMENSION w = downsampled_width;
  JDIMENSION i, i0, tail;
  int inrow, outrow, v;

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    for (v = 0; v < 2; v++) {
      const JSAMPLE *cenp = input_data[inrow];
      const JSAMPLE *nbrp = (v == 0) ? input_data[inrow - 1]
                                     : input_data[inrow + 1];
      JSAMPLE *outptr = output_data[outrow++];
      int thiscolsum, nextcolsum, lastcolsum;

      /* First column has no left neighbour. */
      thiscolsum = GETJSAMPLE(cenp[0]) * 3 + GETJSAMPLE(nbrp[0]);
      nextcolsum = GETJSAMPLE(cenp[1]) * 3 + GETJSAMPLE(nbrp[1]);
      outptr[0] = (JSAMPLE)((thiscolsum * 4 + 8) >> 4);
      outptr[1] = (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);

      tail = 1;
      for (i0 = 0; i0 + 16 < w; i0 += 16) {
        lsx_h2v2_fancy_block(nbrp + i0, cenp + i0, outptr + 2 * i0 + 1);
        tail = i0 + 16;
      }

      for (i = tail; i + 1 < w; i++) {
        lastcolsum = GETJSAMPLE(cenp[i - 1]) * 3 + GETJSAMPLE(nbrp[i - 1]);
        thiscolsum = GETJSAMPLE(cenp[i]) * 3 + GETJSAMPLE(nbrp[i]);
        nextcolsum = GETJSAMPLE(cenp[i + 1]) * 3 + GETJSAMPLE(nbrp[i + 1]);
        outptr[2 * i] = (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
        outptr[2 * i + 1] = (JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
      }

      /* Last column has no right neighbour. */
      lastcolsum = GETJSAMPLE(cenp[w - 2]) * 3 + GETJSAMPLE(nbrp[w - 2]);
      thiscolsum = GETJSAMPLE(cenp[w - 1]) * 3 + GETJSAMPLE(nbrp[w - 1]);
      outptr[2 * w - 2] = (JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
      outptr[2 * w - 1] = (JSAMPLE)((thiscolsum * 4 + 7) >> 4);
    }
    inrow++;
  }
}
/* ==== END fn=jsimd_h2v2_fancy_upsample_lsx ==== */


/* ==== BEGIN fn=jsimd_h1v2_fancy_upsample_lsx ==== */
GLOBAL(void)
jsimd_h1v2_fancy_upsample_lsx(int max_v_samp_factor,
                              JDIMENSION downsampled_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JDIMENSION w = downsampled_width;
  JDIMENSION i, i0;
  int inrow, outrow, v;

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    for (v = 0; v < 2; v++) {
      const JSAMPLE *cenp = input_data[inrow];
      const JSAMPLE *nbrp = (v == 0) ? input_data[inrow - 1]
                                     : input_data[inrow + 1];
      JSAMPLE *outptr = output_data[outrow++];
      int bias = (v == 0) ? 1 : 2;

      i0 = 0;
      for (; i0 + 16 <= w; i0 += 16)
        lsx_h1v2_fancy_block(cenp + i0, nbrp + i0, outptr + i0, bias);

      for (i = i0; i < w; i++) {
        int thiscolsum = GETJSAMPLE(cenp[i]) * 3 + GETJSAMPLE(nbrp[i]);
        outptr[i] = (JSAMPLE)((thiscolsum + bias) >> 2);
      }
    }
    inrow++;
  }
}
/* ==== END fn=jsimd_h1v2_fancy_upsample_lsx ==== */


/* ==== BEGIN fn=jsimd_h2v1_upsample_lsx ==== */
GLOBAL(void)
jsimd_h2v1_upsample_lsx(int max_v_samp_factor, JDIMENSION output_width,
                        JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JDIMENSION i0, colctr;
  int inrow;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    const JSAMPLE *inptr = input_data[inrow];
    JSAMPLE *outptr = output_data[inrow];

    for (i0 = 0; 2 * (i0 + 16) <= output_width; i0 += 16)
      lsx_dup_store(__lsx_vld(inptr + i0, 0), outptr + 2 * i0);

    for (colctr = 2 * i0; colctr < output_width; colctr++)
      outptr[colctr] = inptr[colctr >> 1];
  }
}
/* ==== END fn=jsimd_h2v1_upsample_lsx ==== */


/* ==== BEGIN fn=jsimd_h2v2_upsample_lsx ==== */
GLOBAL(void)
jsimd_h2v2_upsample_lsx(int max_v_samp_factor, JDIMENSION output_width,
                        JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JDIMENSION i0, colctr;
  int inrow, outrow;

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    const JSAMPLE *inptr = input_data[inrow];
    JSAMPLE *outptr0 = output_data[outrow++];
    JSAMPLE *outptr1 = output_data[outrow++];

    for (i0 = 0; 2 * (i0 + 16) <= output_width; i0 += 16) {
      __m128i s = __lsx_vld(inptr + i0, 0);
      lsx_dup_store(s, outptr0 + 2 * i0);
      lsx_dup_store(s, outptr1 + 2 * i0);
    }

    for (colctr = 2 * i0; colctr < output_width; colctr++) {
      JSAMPLE v = inptr[colctr >> 1];
      outptr0[colctr] = v;
      outptr1[colctr] = v;
    }
    inrow++;
  }
}
/* ==== END fn=jsimd_h2v2_upsample_lsx ==== */