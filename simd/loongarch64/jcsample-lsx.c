/*
 * jcsample-lsx.c -- LoongArch LSX implementations of the 2:1 horizontal
 *                   subsamplers used by the encoder (jcsample.c).
 *
 * Numerically identical to h2v1_downsample() / h2v2_downsample(), including
 * the ordered-dither bias (0,1,0,1,... resp. 1,2,1,2,...).  The C versions
 * call expand_right_edge() first and then read past image_width; here the
 * tail is handled with the same clamp (last valid pixel repeated) without
 * touching the caller's buffer.
 *
 * Core trick: 16 input bytes packed as eight u16 lanes already hold the
 * adjacent pixel pairs in their low/high byte, so one AND plus one logical
 * shift produces all eight pairwise sums without any shuffling.  Verified
 * with sprobe.c on box185 (gcc 8.3).
 *
 * Copyright (C) 2026, Loongson Technology Corporation Limited.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "lsxint.h"

/* bias patterns, one lane per output sample of a group of eight */
static const unsigned short k_bias_h2v1[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
static const unsigned short k_bias_h2v2[8] = { 1, 2, 1, 2, 1, 2, 1, 2 };

/* ==== BEGIN fn=lsx_pair_sums ==== */
/* Sixteen u8 pixels -> eight u16 sums of adjacent pixels.  LSX is
 * little-endian, so the byte at the lower address is the low byte of the
 * u16 lane. */
static INLINE __m128i
lsx_pair_sums(__m128i v)
{
  const __m128i mask = __lsx_vreplgr2vr_h(0x00FF);
  return __lsx_vadd_h(__lsx_vand_v(v, mask), __lsx_vsrli_h(v, 8));
}
/* ==== END fn=lsx_pair_sums ==== */

/* ==== BEGIN fn=lsx_downsample_tail ==== */
/* Scalar tail for the last DCT block(s): replicates the last valid pixel
 * exactly like expand_right_edge() does, then applies the dither bias.
 * vsum is 1 for h2v1 and 2 for h2v2 (the number of rows summed). */
static void
lsx_downsample_tail(JSAMPROW in0, JSAMPROW in1, JSAMPROW out,
                    JDIMENSION first_col, JDIMENSION output_cols,
                    JDIMENSION image_width, int vsum)
{
  JDIMENSION c;

  for (c = first_col; c < output_cols; c++) {
    JDIMENSION a = 2 * c, b = a + 1;
    JDIMENSION ai = a < image_width ? a : image_width - 1;
    JDIMENSION bi = b < image_width ? b : image_width - 1;
    int sum = in0[ai] + in0[bi];
    int bias = (int) (c & 1);

    if (vsum == 2) {
      sum += in1[ai] + in1[bi];
      bias += 1;
      out[c] = (JSAMPLE) ((sum + bias) >> 2);
    } else {
      out[c] = (JSAMPLE) ((sum + bias) >> 1);
    }
  }
}
/* ==== END fn=lsx_downsample_tail ==== */

/* ==== BEGIN fn=jsimd_h2v1_downsample_lsx ==== */
GLOBAL(void)
jsimd_h2v1_downsample_lsx(JDIMENSION image_width, int max_v_samp_factor,
                          JDIMENSION v_samp_factor,
                          JDIMENSION width_in_blocks,
                          JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  JDIMENSION full = image_width / (2 * DCTSIZE);
  JDIMENSION outrow;
  int i;

  (void) max_v_samp_factor;
  if (full > width_in_blocks)
    full = width_in_blocks;

  for (outrow = 0; outrow < v_samp_factor; outrow++) {
    JSAMPROW inptr = input_data[outrow];
    JSAMPROW outptr = output_data[outrow];
    const __m128i bias = __lsx_vld(k_bias_h2v1, 0);

    for (i = 0; i < (int) full; i++) {
      __m128i sums = lsx_pair_sums(__lsx_vld(inptr + i * 2 * DCTSIZE, 0));

      sums = __lsx_vadd_h(sums, bias);
      /* only the low eight bytes are valid: never write 16 */
      __lsx_vstelm_d(__lsx_vsrlni_b_h(sums, sums, 1), outptr + i * DCTSIZE,
                     0, 0);
    }
    if (full < width_in_blocks)
      lsx_downsample_tail(inptr, NULL, outptr, full * DCTSIZE, output_cols,
                          image_width, 1);
  }
}
/* ==== END fn=jsimd_h2v1_downsample_lsx ==== */

/* ==== BEGIN fn=jsimd_h2v2_downsample_lsx ==== */
GLOBAL(void)
jsimd_h2v2_downsample_lsx(JDIMENSION image_width, int max_v_samp_factor,
                          JDIMENSION v_samp_factor,
                          JDIMENSION width_in_blocks,
                          JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  JDIMENSION full = image_width / (2 * DCTSIZE);
  JDIMENSION outrow;
  int i;

  (void) max_v_samp_factor;
  if (full > width_in_blocks)
    full = width_in_blocks;

  for (outrow = 0; outrow < v_samp_factor; outrow++) {
    JSAMPROW inptr0 = input_data[outrow * 2];
    JSAMPROW inptr1 = input_data[outrow * 2 + 1];
    JSAMPROW outptr = output_data[outrow];
    const __m128i bias = __lsx_vld(k_bias_h2v2, 0);

    for (i = 0; i < (int) full; i++) {
      __m128i s0 = lsx_pair_sums(__lsx_vld(inptr0 + i * 2 * DCTSIZE, 0));
      __m128i s1 = lsx_pair_sums(__lsx_vld(inptr1 + i * 2 * DCTSIZE, 0));
      __m128i sums = __lsx_vadd_h(__lsx_vadd_h(s0, s1), bias);

      __lsx_vstelm_d(__lsx_vsrlni_b_h(sums, sums, 2), outptr + i * DCTSIZE,
                     0, 0);
    }
    if (full < width_in_blocks)
      lsx_downsample_tail(inptr0, inptr1, outptr, full * DCTSIZE, output_cols,
                          image_width, 2);
  }
}
/* ==== END fn=jsimd_h2v2_downsample_lsx ==== */
