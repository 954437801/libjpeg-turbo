/*
 * jsimd_loongarch64.c
 *
 * Interface between the "normal" portions of the library and the SIMD
 * implementations when running on a 64-bit LoongArch architecture (LSX).
 *
 * All jsimd_* entry points must be defined here because the C code takes
 * their addresses unconditionally; entry points whose jsimd_can_*()
 * counterpart returns 0 are never actually called.
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
#include "jconfigint.h"

#include <stdio.h>
#include <string.h>
#include <sys/auxv.h>

#ifndef HWCAP_LOONGARCH_LSX
#define HWCAP_LOONGARCH_LSX (1 << 4)
#endif

static unsigned int simd_support = ~0;

/* ==== BEGIN fn=init_simd ==== */
LOCAL(void)
init_simd(void)
{
  if (simd_support != ~0U)
    return;

  simd_support = 0;
  if (getauxval(AT_HWCAP) & HWCAP_LOONGARCH_LSX)
    simd_support |= JSIMD_LSX;

#ifndef NO_GETENV
  {
    char *env = getenv("JSIMD_NOLSX");
    if ((env != NULL) && (strcmp(env, "1") == 0))
      simd_support = 0;
    env = getenv("JSIMD_FORCELSX");
    if ((env != NULL) && (strcmp(env, "1") == 0))
      simd_support = JSIMD_LSX;
  }
#endif
}
/* ==== END fn=init_simd ==== */

/* ==== BEGIN fn=jpeg_simd_cpu_support ==== */
GLOBAL(unsigned int)
jpeg_simd_cpu_support(void)
{
  init_simd();
  return simd_support;
}
/* ==== END fn=jpeg_simd_cpu_support ==== */

/* ==== BEGIN fn=jsimd_c_can_null_convert ==== */
GLOBAL(int)
jsimd_c_can_null_convert(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_c_can_null_convert ==== */

/* ==== BEGIN fn=jsimd_can_convsamp ==== */
GLOBAL(int)
jsimd_can_convsamp(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  if (simd_support & JSIMD_LSX)
    return 1;

  return 0;
}
/* ==== END fn=jsimd_can_convsamp ==== */

/* ==== BEGIN fn=jsimd_can_convsamp_float ==== */
GLOBAL(int)
jsimd_can_convsamp_float(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_convsamp_float ==== */

/* ==== BEGIN fn=jsimd_can_encode_mcu_AC_first_prepare ==== */
GLOBAL(int)
jsimd_can_encode_mcu_AC_first_prepare(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_encode_mcu_AC_first_prepare ==== */

/* ==== BEGIN fn=jsimd_can_encode_mcu_AC_refine_prepare ==== */
GLOBAL(int)
jsimd_can_encode_mcu_AC_refine_prepare(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_encode_mcu_AC_refine_prepare ==== */

/* ==== BEGIN fn=jsimd_can_fdct_float ==== */
GLOBAL(int)
jsimd_can_fdct_float(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_fdct_float ==== */

/* ==== BEGIN fn=jsimd_can_fdct_ifast ==== */
GLOBAL(int)
jsimd_can_fdct_ifast(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_fdct_ifast ==== */

/* ==== BEGIN fn=jsimd_can_fdct_islow ==== */
GLOBAL(int)
jsimd_can_fdct_islow(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_fdct_islow ==== */

/* ==== BEGIN fn=jsimd_can_h1v2_fancy_upsample ==== */
GLOBAL(int)
jsimd_can_h1v2_fancy_upsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h1v2_fancy_upsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v1_downsample ==== */
GLOBAL(int)
jsimd_can_h2v1_downsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v1_downsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v1_fancy_upsample ==== */
GLOBAL(int)
jsimd_can_h2v1_fancy_upsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v1_fancy_upsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v1_merged_upsample ==== */
GLOBAL(int)
jsimd_can_h2v1_merged_upsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v1_merged_upsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v1_upsample ==== */
GLOBAL(int)
jsimd_can_h2v1_upsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v1_upsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v2_downsample ==== */
GLOBAL(int)
jsimd_can_h2v2_downsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v2_downsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v2_fancy_upsample ==== */
GLOBAL(int)
jsimd_can_h2v2_fancy_upsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v2_fancy_upsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v2_merged_upsample ==== */
GLOBAL(int)
jsimd_can_h2v2_merged_upsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v2_merged_upsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v2_smooth_downsample ==== */
GLOBAL(int)
jsimd_can_h2v2_smooth_downsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v2_smooth_downsample ==== */

/* ==== BEGIN fn=jsimd_can_h2v2_upsample ==== */
GLOBAL(int)
jsimd_can_h2v2_upsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_h2v2_upsample ==== */

/* ==== BEGIN fn=jsimd_can_huff_encode_one_block ==== */
GLOBAL(int)
jsimd_can_huff_encode_one_block(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_huff_encode_one_block ==== */

/* ==== BEGIN fn=jsimd_can_idct_12x12 ==== */
GLOBAL(int)
jsimd_can_idct_12x12(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_idct_12x12 ==== */

/* ==== BEGIN fn=jsimd_can_idct_2x2 ==== */
GLOBAL(int)
jsimd_can_idct_2x2(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_idct_2x2 ==== */

/* ==== BEGIN fn=jsimd_can_idct_4x4 ==== */
GLOBAL(int)
jsimd_can_idct_4x4(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_idct_4x4 ==== */

/* ==== BEGIN fn=jsimd_can_idct_6x6 ==== */
GLOBAL(int)
jsimd_can_idct_6x6(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_idct_6x6 ==== */

/* ==== BEGIN fn=jsimd_can_idct_float ==== */
GLOBAL(int)
jsimd_can_idct_float(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_idct_float ==== */

/* ==== BEGIN fn=jsimd_can_idct_ifast ==== */
GLOBAL(int)
jsimd_can_idct_ifast(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_idct_ifast ==== */

/* ==== BEGIN fn=jsimd_can_idct_islow ==== */
GLOBAL(int)
jsimd_can_idct_islow(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_idct_islow ==== */

/* ==== BEGIN fn=jsimd_can_int_upsample ==== */
GLOBAL(int)
jsimd_can_int_upsample(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_int_upsample ==== */

/* ==== BEGIN fn=jsimd_can_quantize ==== */
GLOBAL(int)
jsimd_can_quantize(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  if (simd_support & JSIMD_LSX)
    return 1;

  return 0;
}
/* ==== END fn=jsimd_can_quantize ==== */

/* ==== BEGIN fn=jsimd_can_quantize_float ==== */
GLOBAL(int)
jsimd_can_quantize_float(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_quantize_float ==== */

/* ==== BEGIN fn=jsimd_can_rgb_gray ==== */
GLOBAL(int)
jsimd_can_rgb_gray(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_rgb_gray ==== */

/* ==== BEGIN fn=jsimd_can_rgb_ycc ==== */
GLOBAL(int)
jsimd_can_rgb_ycc(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return 0;

  if (simd_support & JSIMD_LSX)
    return 1;

  return 0;
}
/* ==== END fn=jsimd_can_rgb_ycc ==== */

/* ==== BEGIN fn=jsimd_can_ycc_rgb ==== */
GLOBAL(int)
jsimd_can_ycc_rgb(void)
{
  init_simd();

  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return 0;

  if (simd_support & JSIMD_LSX)
    return 1;

  return 0;
}
/* ==== END fn=jsimd_can_ycc_rgb ==== */

/* ==== BEGIN fn=jsimd_can_ycc_rgb565 ==== */
GLOBAL(int)
jsimd_can_ycc_rgb565(void)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_can_ycc_rgb565 ==== */

/* ==== BEGIN fn=jsimd_c_null_convert ==== */
GLOBAL(void)
jsimd_c_null_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_c_null_convert ==== */

/* ==== BEGIN fn=jsimd_convsamp ==== */
GLOBAL(void)
jsimd_convsamp(JSAMPARRAY sample_data, JDIMENSION start_col, DCTELEM *workspace)
{
  jsimd_convsamp_lsx(sample_data, start_col, workspace);
}
/* ==== END fn=jsimd_convsamp ==== */

/* ==== BEGIN fn=jsimd_convsamp_float ==== */
GLOBAL(void)
jsimd_convsamp_float(JSAMPARRAY sample_data, JDIMENSION start_col, FAST_FLOAT *workspace)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_convsamp_float ==== */

/* ==== BEGIN fn=jsimd_encode_mcu_AC_first_prepare ==== */
GLOBAL(void)
jsimd_encode_mcu_AC_first_prepare(const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al, JCOEF *values, size_t *zerobits)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_encode_mcu_AC_first_prepare ==== */

/* ==== BEGIN fn=jsimd_encode_mcu_AC_refine_prepare ==== */
GLOBAL(int)
jsimd_encode_mcu_AC_refine_prepare(const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al, JCOEF *absvalues, size_t *bits)
{
  /* not implemented on LoongArch yet: fall back to the C version */
  return 0;
}
/* ==== END fn=jsimd_encode_mcu_AC_refine_prepare ==== */

/* ==== BEGIN fn=jsimd_fdct_float ==== */
GLOBAL(void)
jsimd_fdct_float(FAST_FLOAT *data)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_fdct_float ==== */

/* ==== BEGIN fn=jsimd_fdct_ifast ==== */
GLOBAL(void)
jsimd_fdct_ifast(DCTELEM *data)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_fdct_ifast ==== */

/* ==== BEGIN fn=jsimd_fdct_islow ==== */
GLOBAL(void)
jsimd_fdct_islow(DCTELEM *data)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_fdct_islow ==== */

/* ==== BEGIN fn=jsimd_h1v2_fancy_upsample ==== */
GLOBAL(void)
jsimd_h1v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h1v2_fancy_upsample ==== */

/* ==== BEGIN fn=jsimd_h2v1_downsample ==== */
GLOBAL(void)
jsimd_h2v1_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v1_downsample ==== */

/* ==== BEGIN fn=jsimd_h2v1_fancy_upsample ==== */
GLOBAL(void)
jsimd_h2v1_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v1_fancy_upsample ==== */

/* ==== BEGIN fn=jsimd_h2v1_merged_upsample ==== */
GLOBAL(void)
jsimd_h2v1_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v1_merged_upsample ==== */

/* ==== BEGIN fn=jsimd_h2v1_upsample ==== */
GLOBAL(void)
jsimd_h2v1_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v1_upsample ==== */

/* ==== BEGIN fn=jsimd_h2v2_downsample ==== */
GLOBAL(void)
jsimd_h2v2_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v2_downsample ==== */

/* ==== BEGIN fn=jsimd_h2v2_fancy_upsample ==== */
GLOBAL(void)
jsimd_h2v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v2_fancy_upsample ==== */

/* ==== BEGIN fn=jsimd_h2v2_merged_upsample ==== */
GLOBAL(void)
jsimd_h2v2_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf, JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v2_merged_upsample ==== */

/* ==== BEGIN fn=jsimd_h2v2_smooth_downsample ==== */
GLOBAL(void)
jsimd_h2v2_smooth_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v2_smooth_downsample ==== */

/* ==== BEGIN fn=jsimd_h2v2_upsample ==== */
GLOBAL(void)
jsimd_h2v2_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_h2v2_upsample ==== */

/* ==== BEGIN fn=jsimd_huff_encode_one_block ==== */
GLOBAL(JOCTET *)
jsimd_huff_encode_one_block(void *state, JOCTET *buffer, JCOEFPTR block, int last_dc_val, c_derived_tbl *dctbl, c_derived_tbl *actbl)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  return NULL;
}
/* ==== END fn=jsimd_huff_encode_one_block ==== */

/* ==== BEGIN fn=jsimd_idct_12x12 ==== */
GLOBAL(void)
jsimd_idct_12x12(j_decompress_ptr cinfo, jpeg_component_info *compptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_idct_12x12 ==== */

/* ==== BEGIN fn=jsimd_idct_2x2 ==== */
GLOBAL(void)
jsimd_idct_2x2(j_decompress_ptr cinfo, jpeg_component_info *compptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_idct_2x2 ==== */

/* ==== BEGIN fn=jsimd_idct_4x4 ==== */
GLOBAL(void)
jsimd_idct_4x4(j_decompress_ptr cinfo, jpeg_component_info *compptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_idct_4x4 ==== */

/* ==== BEGIN fn=jsimd_idct_6x6 ==== */
GLOBAL(void)
jsimd_idct_6x6(j_decompress_ptr cinfo, jpeg_component_info *compptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_idct_6x6 ==== */

/* ==== BEGIN fn=jsimd_idct_float ==== */
GLOBAL(void)
jsimd_idct_float(j_decompress_ptr cinfo, jpeg_component_info *compptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_idct_float ==== */

/* ==== BEGIN fn=jsimd_idct_ifast ==== */
GLOBAL(void)
jsimd_idct_ifast(j_decompress_ptr cinfo, jpeg_component_info *compptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_idct_ifast ==== */

/* ==== BEGIN fn=jsimd_idct_islow ==== */
GLOBAL(void)
jsimd_idct_islow(j_decompress_ptr cinfo, jpeg_component_info *compptr, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_idct_islow ==== */

/* ==== BEGIN fn=jsimd_int_upsample ==== */
GLOBAL(void)
jsimd_int_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr, JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_int_upsample ==== */

/* ==== BEGIN fn=jsimd_quantize ==== */
GLOBAL(void)
jsimd_quantize(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
  jsimd_quantize_lsx(coef_block, divisors, workspace);
}
/* ==== END fn=jsimd_quantize ==== */

/* ==== BEGIN fn=jsimd_quantize_float ==== */
GLOBAL(void)
jsimd_quantize_float(JCOEFPTR coef_block, FAST_FLOAT *divisors, FAST_FLOAT *workspace)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_quantize_float ==== */

/* ==== BEGIN fn=jsimd_rgb_gray_convert ==== */
GLOBAL(void)
jsimd_rgb_gray_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_rgb_gray_convert ==== */

/* ==== BEGIN fn=jsimd_rgb_ycc_convert ==== */
GLOBAL(void)
jsimd_rgb_ycc_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf, JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  jsimd_rgb_ycc_convert_lsx(cinfo, input_buf, output_buf, output_row, num_rows);
}
/* ==== END fn=jsimd_rgb_ycc_convert ==== */

/* ==== BEGIN fn=jsimd_ycc_rgb565_convert ==== */
GLOBAL(void)
jsimd_ycc_rgb565_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf, JDIMENSION input_row, JSAMPARRAY output_buf, int num_rows)
{
  /* not implemented on LoongArch yet: this path is never taken
   * because the corresponding jsimd_can_*() returns 0. */
  (void)0;
}
/* ==== END fn=jsimd_ycc_rgb565_convert ==== */

/* ==== BEGIN fn=jsimd_ycc_rgb_convert ==== */
GLOBAL(void)
jsimd_ycc_rgb_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf, JDIMENSION input_row, JSAMPARRAY output_buf, int num_rows)
{
  jsimd_ycc_rgb_convert_lsx(cinfo, input_buf, input_row, output_buf, num_rows);
}
/* ==== END fn=jsimd_ycc_rgb_convert ==== */
