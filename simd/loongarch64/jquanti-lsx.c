/*
 * jquanti-lsx.c
 *
 * Sample data conversion (convsamp) and coefficient quantization,
 * LoongArch LSX (128-bit) SIMD.
 *
 * convsamp() subtracts CENTERJSAMPLE and widens 8-bit samples to 16-bit,
 * filling one 8x8 DCT workspace block per call.
 *
 * quantize() divides the DCT coefficients by the quantization table without
 * using a division: the divisor table built by compute_reciprocal() in
 * jcdctmgr.c already holds, per coefficient,
 *     recip = floor(2^r / divisor)        (unsigned 16-bit)
 *     corr  = divisor / 2  (+1)           (unsigned 16-bit, rounding term)
 *     shift = r - sizeof(DCTELEM) * 8     (signed 16-bit)
 * so that   q = (|t| + corr) * recip >> (shift + 16)
 * reproduces the rounded quotient, after which the sign is restored.
 *
 * Numerical output is bit for bit identical to quantize() in jcdctmgr.c:
 * the product is kept 32-bit and shifted in a single step, and the 16-bit
 * narrowing is a plain truncation (vsrarni, no rounding at ui5=0, no
 * saturation), exactly like the C cast to DCTELEM.
 *
 * Copyright (C) 2026, Loongson Technology Corporation Limited.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"
#include "../../jsimddct.h"
#include "../jsimd.h"
#include "lsxint.h"

/* ==== BEGIN fn=jsimd_convsamp_lsx ==== */
/* Load 8 samples of one row, subtract CENTERJSAMPLE and widen to 16-bit.
 * vldrepl_d reads exactly 8 bytes (the low half of the register holds the
 * row slice), so unlike a 16-byte vld it never reads past the end of the
 * sample row. */
GLOBAL(void)
jsimd_convsamp_lsx(JSAMPARRAY sample_data, JDIMENSION start_col,
                   DCTELEM *workspace)
{
  const __m128i zero = __lsx_vreplgr2vr_b(0);
  const __m128i c128 = __lsx_vreplgr2vr_h(CENTERJSAMPLE);
  int i;

  for (i = 0; i < DCTSIZE; i++) {
    __m128i raw = __lsx_vldrepl_d(sample_data[i] + start_col, 0);
    __m128i s16 = __lsx_vilvl_b(zero, raw);      /* u8 -> u16, low 8 valid */
    __m128i d16 = __lsx_vsub_h(s16, c128);
    __lsx_vst(d16, workspace + i * DCTSIZE, 0);
  }
}
/* ==== END fn=jsimd_convsamp_lsx ==== */

/* ==== BEGIN fn=jsimd_quantize_lsx ==== */
GLOBAL(void)
jsimd_quantize_lsx(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
  /* Table layout produced by compute_reciprocal() in jcdctmgr.c: */
  UDCTELEM *recip_ptr = (UDCTELEM *)divisors;             /* [0  .. 63] */
  UDCTELEM *corr_ptr = (UDCTELEM *)divisors + DCTSIZE2;   /* [64 ..127] */
  DCTELEM *shift_ptr = divisors + 3 * DCTSIZE2;           /* [192..255] */
  int i;

  for (i = 0; i < DCTSIZE2; i += DCTSIZE) {
    __m128i x = __lsx_vld(workspace + i, 0);          /* 8 x int16 */
    __m128i recip = __lsx_vld(recip_ptr + i, 0);      /* 8 x uint16 */
    __m128i corr = __lsx_vld(corr_ptr + i, 0);        /* 8 x uint16 */
    __m128i sh = __lsx_vld(shift_ptr + i, 0);         /* 8 x int16 */

    /* |x| via the branchless (x ^ sign) - sign, sign = x >> 15.
     * The same expression restores the sign further down. */
    __m128i sign = __lsx_vsrai_h(x, 15);
    __m128i ax = __lsx_vsub_h(__lsx_vxor_v(x, sign), sign);

    /* |x| + corr as unsigned 16-bit.  corr is divisor/2 and the divisor is
     * quantval << 3 (<= 2040 for an 8-bit JPEG), so the sum stays below
     * 65534 and cannot wrap -- matching C's int-promoted addition. */
    __m128i sum = __lsx_vadd_h(ax, corr);

    /* Widening unsigned multiply: the sum can exceed 32767, so the signed
     * variant would be wrong.  Even/odd halfword products are then
     * re-interleaved into natural order (vilvl/vilvh put the second operand
     * first, hence po as the first argument). */
    __m128i pe = __lsx_vmulwev_w_hu(sum, recip);       /* p0 p2 p4 p6 */
    __m128i po = __lsx_vmulwod_w_hu(sum, recip);       /* p1 p3 p5 p7 */
    __m128i plo = __lsx_vilvl_w(po, pe);               /* p0 p1 p2 p3 */
    __m128i phi = __lsx_vilvh_w(po, pe);               /* p4 p5 p6 p7 */

    /* Per-element shift by (shift[i] + 16).  The table holds a signed
     * 16-bit value, so it has to be widened to 32-bit before vsrl_w. */
    __m128i n16 = __lsx_vreplgr2vr_w(16);
    __m128i slo = __lsx_vadd_w(__lsx_vsllwil_w_h(sh, 0), n16);
    __m128i shi = __lsx_vadd_w(__lsx_vexth_w_h(sh), n16);
    plo = __lsx_vsrl_w(plo, slo);
    phi = __lsx_vsrl_w(phi, shi);

    /* Narrow back to 16-bit.  vsrarni with ui5=0 truncates without
     * rounding and without saturation, so this equals C's (DCTELEM) cast;
     * it writes narrow(plo) into the low half and narrow(phi) into the high
     * half of the same register. */
    __m128i mag = __lsx_vsrarni_h_w(phi, plo, 0);

    /* Restore the sign of the original coefficient. */
    __m128i out = __lsx_vsub_h(__lsx_vxor_v(mag, sign), sign);

    __lsx_vst(out, coef_block + i, 0);
  }
}
/* ==== END fn=jsimd_quantize_lsx ==== */
