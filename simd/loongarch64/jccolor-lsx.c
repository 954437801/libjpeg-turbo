/*
 * jccolor-lsx.c
 *
 * RGB(A) -> YCbCr color conversion, LoongArch LSX (128-bit) SIMD.
 *
 * Covers JCS_RGB (3 bytes, the path Qt's qjpeg takes) as well as
 * JCS_EXT_RGB / EXT_BGR / EXT_RGBX / EXT_BGRX / EXT_XBGR / EXT_XRGB.
 * Numerical output is byte for byte identical to the C implementation in
 * jccolor.c.
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

#include <stdio.h>

/* 3-byte input sampling indices (RGB order; used for JCS_RGB and
 * JCS_EXT_RGB alike) */
static const unsigned char k_ridx3[16] = { 0, 3, 6, 9, 12, 15, 18, 21, 0, 0, 0, 0, 0, 0, 0, 0 };
static const unsigned char k_gidx3[16] = { 1, 4, 7, 10, 13, 16, 19, 22, 0, 0, 0, 0, 0, 0, 0, 0 };
static const unsigned char k_bidx3[16] = { 2, 5, 8, 11, 14, 17, 20, 23, 0, 0, 0, 0, 0, 0, 0, 0 };

/* ==== BEGIN fn=lsx_rgb_ycc_core8 ==== */
/* 8 pixels of R/G/B (each split into lo/hi v4i32) -> Y/Cb/Cr, 8 bytes each */
static INLINE void
lsx_rgb_ycc_core8(__m128i r_lo, __m128i r_hi, __m128i g_lo, __m128i g_hi,
                  __m128i b_lo, __m128i b_hi,
                  const __m128i *kYR, const __m128i *kYG, const __m128i *kYB,
                  const __m128i *kCbR, const __m128i *kCbG, const __m128i *kCbB,
                  const __m128i *kCrR, const __m128i *kCrG, const __m128i *kCrB,
                  const __m128i *half, const __m128i *bias,
                  JSAMPLE *oy, JSAMPLE *ocb, JSAMPLE *ocr)
{
  __m128i y_lo = __lsx_vsrai_w(__lsx_vadd_w(__lsx_vadd_w(
      __lsx_vmul_w(*kYR, r_lo), __lsx_vmul_w(*kYG, g_lo)),
      __lsx_vadd_w(__lsx_vmul_w(*kYB, b_lo), *half)), 16);
  __m128i y_hi = __lsx_vsrai_w(__lsx_vadd_w(__lsx_vadd_w(
      __lsx_vmul_w(*kYR, r_hi), __lsx_vmul_w(*kYG, g_hi)),
      __lsx_vadd_w(__lsx_vmul_w(*kYB, b_hi), *half)), 16);
  __m128i cb_lo = __lsx_vsrai_w(__lsx_vadd_w(__lsx_vadd_w(
      __lsx_vmul_w(*kCbR, r_lo), __lsx_vmul_w(*kCbG, g_lo)),
      __lsx_vadd_w(__lsx_vmul_w(*kCbB, b_lo), *bias)), 16);
  __m128i cb_hi = __lsx_vsrai_w(__lsx_vadd_w(__lsx_vadd_w(
      __lsx_vmul_w(*kCbR, r_hi), __lsx_vmul_w(*kCbG, g_hi)),
      __lsx_vadd_w(__lsx_vmul_w(*kCbB, b_hi), *bias)), 16);
  __m128i cr_lo = __lsx_vsrai_w(__lsx_vadd_w(__lsx_vadd_w(
      __lsx_vmul_w(*kCrR, r_lo), __lsx_vmul_w(*kCrG, g_lo)),
      __lsx_vadd_w(__lsx_vmul_w(*kCrB, b_lo), *bias)), 16);
  __m128i cr_hi = __lsx_vsrai_w(__lsx_vadd_w(__lsx_vadd_w(
      __lsx_vmul_w(*kCrR, r_hi), __lsx_vmul_w(*kCrG, g_hi)),
      __lsx_vadd_w(__lsx_vmul_w(*kCrB, b_hi), *bias)), 16);

  {
    union { __m128i v; JSAMPLE b[16]; } u;
    u.v = lsx_pack8(y_lo, y_hi);   memcpy(oy,  u.b, 8);
    u.v = lsx_pack8(cb_lo, cb_hi); memcpy(ocb, u.b, 8);
    u.v = lsx_pack8(cr_lo, cr_hi); memcpy(ocr, u.b, 8);
  }
}
/* ==== END fn=lsx_rgb_ycc_core8 ==== */

/* ==== BEGIN fn=jsimd_rgb_ycc_convert_lsx ==== */
/* Coefficients match jccolor.c:
 *   Y  = (19595 R + 38470 G + 7471 B + 32768) >> 16
 *   Cb = (-11059 R - 21709 G + 32768 B + 8421375) >> 16
 *   Cr = (32768 R - 27439 G - 5329 B + 8421375) >> 16
 * 4-byte pixels are extracted with vsrl_w + vand (no byte shuffling).
 */
void
jsimd_rgb_ycc_convert_lsx(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                          JSAMPIMAGE output_buf, JDIMENSION output_row,
                          int num_rows)
{
  const JDIMENSION width = cinfo->image_width;
  const __m128i mask = __lsx_vreplgr2vr_w(0xff);
  const __m128i kYR = __lsx_vreplgr2vr_w(19595);
  const __m128i kYG = __lsx_vreplgr2vr_w(38470);
  const __m128i kYB = __lsx_vreplgr2vr_w(7471);
  const __m128i kCbR = __lsx_vreplgr2vr_w(-11059);
  const __m128i kCbG = __lsx_vreplgr2vr_w(-21709);
  const __m128i kCbB = __lsx_vreplgr2vr_w(32768);
  const __m128i kCrR = __lsx_vreplgr2vr_w(32768);
  const __m128i kCrG = __lsx_vreplgr2vr_w(-27439);
  const __m128i kCrB = __lsx_vreplgr2vr_w(-5329);
  const __m128i half = __lsx_vreplgr2vr_w(32768);
  /* CBCR_OFFSET + ONE_HALF - 1 = 8388608 + 32768 - 1.  The -1 is jccolor.c's
   * "0.5-epsilon" rounding fudge: it keeps the maximum output at MAXJSAMPLE
   * instead of MAXJSAMPLE+1, so no range-limiting is needed. */
  const __m128i bias = __lsx_vreplgr2vr_w(8421375);
  int shR, shG, shB, pixsize;
  __m128i vshR, vshG, vshB;

  pixsize = (cinfo->in_color_space == JCS_RGB || cinfo->in_color_space == JCS_EXT_RGB ||
             cinfo->in_color_space == JCS_EXT_BGR) ? 3 : 4;

  switch (cinfo->in_color_space) {
  case JCS_EXT_BGRX: shB = 0;  shG = 8;  shR = 16; break;
  case JCS_EXT_RGBX: shB = 16; shG = 8;  shR = 0;  break;
  case JCS_EXT_XBGR: shB = 8;  shG = 16; shR = 24; break;
  default:           shB = 24; shG = 16; shR = 8;  break;  /* XRGB */
  }
  /* vsrli_w requires an immediate, so broadcast the shift amount and use
   * the variable-shift form */
  vshR = __lsx_vreplgr2vr_w(shR);
  vshG = __lsx_vreplgr2vr_w(shG);
  vshB = __lsx_vreplgr2vr_w(shB);

#ifdef LSX_DEBUG
  fprintf(stderr, "[LSX] rgb_ycc_convert: w=%u rows=%d cs=%d\n",
          (unsigned)width, num_rows, (int)cinfo->in_color_space);
#endif

  while (num_rows-- > 0) {
    const JSAMPLE *in = input_buf[0];
    JSAMPLE *oy  = output_buf[0][output_row];
    JSAMPLE *ocb = output_buf[1][output_row];
    JSAMPLE *ocr = output_buf[2][output_row];
    JDIMENSION col = 0;
    input_buf++;
    output_row++;

    for (; col + 8 <= width; col += 8) {
      if (pixsize == 3) {
        /* 3-byte input: copy into a 32-byte stack buffer first (avoids
         * reading past the caller's row buffer), then pick R/G/B by index */
        JSAMPLE tmp[32] __attribute__((aligned(16)));
        __m128i v0, v1, r8, g8, b8, r16, g16, b16;
        const __m128i z = __lsx_vreplgr2vr_b(0);
        memcpy(tmp, in, 24);
        v0 = __lsx_vld(tmp, 0);
        v1 = __lsx_vld(tmp + 16, 0);
        r8 = __lsx_vshuf_b(v1, v0, __lsx_vld(k_ridx3, 0));
        g8 = __lsx_vshuf_b(v1, v0, __lsx_vld(k_gidx3, 0));
        b8 = __lsx_vshuf_b(v1, v0, __lsx_vld(k_bidx3, 0));
        r16 = __lsx_vilvl_b(z, r8); g16 = __lsx_vilvl_b(z, g8); b16 = __lsx_vilvl_b(z, b8);
        lsx_rgb_ycc_core8(__lsx_vilvl_h(z, r16), __lsx_vilvh_h(z, r16),
                          __lsx_vilvl_h(z, g16), __lsx_vilvh_h(z, g16),
                          __lsx_vilvl_h(z, b16), __lsx_vilvh_h(z, b16),
                          &kYR, &kYG, &kYB, &kCbR, &kCbG, &kCbB, &kCrR, &kCrG, &kCrB,
                          &half, &bias, oy + col, ocb + col, ocr + col);
        in += 24;
      } else {
        __m128i v0 = __lsx_vld(in, 0), v1 = __lsx_vld(in + 16, 0);
        __m128i r_lo = __lsx_vand_v(__lsx_vsrl_w(v0, vshR), mask);
        __m128i r_hi = __lsx_vand_v(__lsx_vsrl_w(v1, vshR), mask);
        __m128i g_lo = __lsx_vand_v(__lsx_vsrl_w(v0, vshG), mask);
        __m128i g_hi = __lsx_vand_v(__lsx_vsrl_w(v1, vshG), mask);
        __m128i b_lo = __lsx_vand_v(__lsx_vsrl_w(v0, vshB), mask);
        __m128i b_hi = __lsx_vand_v(__lsx_vsrl_w(v1, vshB), mask);
        lsx_rgb_ycc_core8(r_lo, r_hi, g_lo, g_hi, b_lo, b_hi,
                          &kYR, &kYG, &kYB, &kCbR, &kCbG, &kCbB, &kCrR, &kCrG, &kCrB,
                          &half, &bias, oy + col, ocb + col, ocr + col);
        in += 32;
      }
    }
    for (; col < width; col++) {
      int rr, gg, bb;
      if (pixsize == 3) {
        if (cinfo->in_color_space == JCS_EXT_BGR) { bb = in[0]; gg = in[1]; rr = in[2]; }
        else                                      { rr = in[0]; gg = in[1]; bb = in[2]; }
      } else switch (cinfo->in_color_space) {
      case JCS_EXT_BGRX: bb = in[0]; gg = in[1]; rr = in[2]; break;
      case JCS_EXT_RGBX: rr = in[0]; gg = in[1]; bb = in[2]; break;
      case JCS_EXT_XBGR: bb = in[1]; gg = in[2]; rr = in[3]; break;
      default:           rr = in[1]; gg = in[2]; bb = in[3]; break;
      }
      oy[col]  = (JSAMPLE)((19595 * rr + 38470 * gg + 7471 * bb + 32768) >> 16);
      ocb[col] = (JSAMPLE)((-11059 * rr - 21709 * gg + 32768 * bb + 8421375) >> 16);
      ocr[col] = (JSAMPLE)((32768 * rr - 27439 * gg - 5329 * bb + 8421375) >> 16);
      in += pixsize;
    }
  }
}
/* ==== END fn=jsimd_rgb_ycc_convert_lsx ==== */
