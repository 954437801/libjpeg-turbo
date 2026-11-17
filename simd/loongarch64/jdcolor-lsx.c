/*
 * jdcolor-lsx.c
 *
 * YCbCr -> RGB(A) color conversion, LoongArch LSX (128-bit) SIMD.
 *
 * Covers JCS_RGB (3 bytes, the path Qt's qjpeg takes) as well as
 * JCS_EXT_RGB / EXT_BGR / EXT_RGBX / EXT_BGRX / EXT_XBGR / EXT_XRGB.
 * Numerical output is byte for byte identical to the C implementation in
 * jdcolor.c (16.16 fixed point, saturating pack == range_limit).
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

/* ==== BEGIN fn=lsx_ycc_core8 ==== */
/* 8 pixels of YCbCr -> (R,G,B) as three u8 vectors (low 8 bytes valid) */
static INLINE void
lsx_ycc_core8(const JSAMPLE *yp, const JSAMPLE *cbp, const JSAMPLE *crp,
              __m128i *r8, __m128i *g8, __m128i *b8)
{
  const __m128i v128 = __lsx_vreplgr2vr_h(L_CENTER);
  const __m128i half = __lsx_vreplgr2vr_w(L_ONE_HALF);
  const __m128i kR  = __lsx_vreplgr2vr_w(L_FIX(1.40200));
  const __m128i kB  = __lsx_vreplgr2vr_w(L_FIX(1.77200));
  const __m128i kG1 = __lsx_vreplgr2vr_w(L_FIX(0.34414));
  const __m128i kG2 = __lsx_vreplgr2vr_w(L_FIX(0.71414));

  __m128i y16  = lsx_load8_u16(yp);
  __m128i cb16 = __lsx_vsub_h(lsx_load8_u16(cbp), v128);
  __m128i cr16 = __lsx_vsub_h(lsx_load8_u16(crp), v128);

  __m128i y_lo = lsx_sext_lo(y16),   y_hi = lsx_sext_hi(y16);
  __m128i cb_lo = lsx_sext_lo(cb16), cb_hi = lsx_sext_hi(cb16);
  __m128i cr_lo = lsx_sext_lo(cr16), cr_hi = lsx_sext_hi(cr16);

  __m128i r_lo = __lsx_vadd_w(y_lo, __lsx_vsrai_w(__lsx_vadd_w(__lsx_vmul_w(kR, cr_lo), half), 16));
  __m128i r_hi = __lsx_vadd_w(y_hi, __lsx_vsrai_w(__lsx_vadd_w(__lsx_vmul_w(kR, cr_hi), half), 16));
  __m128i b_lo = __lsx_vadd_w(y_lo, __lsx_vsrai_w(__lsx_vadd_w(__lsx_vmul_w(kB, cb_lo), half), 16));
  __m128i b_hi = __lsx_vadd_w(y_hi, __lsx_vsrai_w(__lsx_vadd_w(__lsx_vmul_w(kB, cb_hi), half), 16));
  __m128i g_lo = __lsx_vadd_w(y_lo, __lsx_vsrai_w(
      __lsx_vsub_w(half, __lsx_vadd_w(__lsx_vmul_w(kG1, cb_lo), __lsx_vmul_w(kG2, cr_lo))), 16));
  __m128i g_hi = __lsx_vadd_w(y_hi, __lsx_vsrai_w(
      __lsx_vsub_w(half, __lsx_vadd_w(__lsx_vmul_w(kG1, cb_hi), __lsx_vmul_w(kG2, cr_hi))), 16));

  *r8 = lsx_pack8(r_lo, r_hi);
  *g8 = lsx_pack8(g_lo, g_hi);
  *b8 = lsx_pack8(b_lo, b_hi);
}
/* ==== END fn=lsx_ycc_core8 ==== */

/* ==== BEGIN fn=jsimd_ycc_rgb_convert_lsx ==== */
void
jsimd_ycc_rgb_convert_lsx(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                          JDIMENSION input_row, JSAMPARRAY output_buf,
                          int num_rows)
{
  const JDIMENSION width = cinfo->output_width;
  int mode;
  switch (cinfo->out_color_space) {
  case JCS_EXT_BGRX: mode = 0; break;
  case JCS_EXT_RGBX: mode = 1; break;
  case JCS_EXT_XBGR: mode = 2; break;
  case JCS_EXT_XRGB: mode = 3; break;
  case JCS_EXT_BGR:  mode = 4; break;   /* 3 bytes: B,G,R */
  default:           mode = 5; break;   /* JCS_RGB / JCS_EXT_RGB: 3 bytes R,G,B */
  }

#ifdef LSX_DEBUG
  fprintf(stderr, "[LSX] ycc_rgb_convert: w=%u rows=%d cs=%d\n",
          (unsigned)width, num_rows, (int)cinfo->out_color_space);
#endif

  while (num_rows-- > 0) {
    const JSAMPLE *yp  = input_buf[0][input_row];
    const JSAMPLE *cbp = input_buf[1][input_row];
    const JSAMPLE *crp = input_buf[2][input_row];
    JSAMPLE *out = output_buf[0];
    JDIMENSION col = 0;
    input_row++;
    output_buf++;

    for (; col + 8 <= width; col += 8) {
      __m128i r8, g8, b8, lo, hi;
      lsx_ycc_core8(yp + col, cbp + col, crp + col, &r8, &g8, &b8);
      if (mode >= 4) {
        /* 3-byte output: interleave as BGRX/RGBX, then drop the pad byte */
        lsx_interleave4(r8, g8, b8, mode == 4 ? 0 : 1, &lo, &hi);
        lsx_store3(lo, hi, out);
        out += 24;
      } else {
        lsx_interleave4(r8, g8, b8, mode, &lo, &hi);
        __lsx_vst(lo, out, 0);
        __lsx_vst(hi, out + 16, 0);
        out += 32;
      }
    }
    for (; col < width; col++) {
      int y  = yp[col];
      int cb = cbp[col] - L_CENTER;
      int cr = crp[col] - L_CENTER;
      int r = y + (int)((L_FIX(1.40200) * cr + L_ONE_HALF) >> 16);
      int g = y + (int)((L_ONE_HALF - L_FIX(0.34414) * cb - L_FIX(0.71414) * cr) >> 16);
      int b = y + (int)((L_FIX(1.77200) * cb + L_ONE_HALF) >> 16);
      r = r < 0 ? 0 : (r > 255 ? 255 : r);
      g = g < 0 ? 0 : (g > 255 ? 255 : g);
      b = b < 0 ? 0 : (b > 255 ? 255 : b);
      if (mode >= 4) {
        if (mode == 4) { out[0] = (JSAMPLE)b; out[1] = (JSAMPLE)g; out[2] = (JSAMPLE)r; }
        else           { out[0] = (JSAMPLE)r; out[1] = (JSAMPLE)g; out[2] = (JSAMPLE)b; }
        out += 3;
      } else {
        switch (mode) {
        case 0: out[0] = (JSAMPLE)b; out[1] = (JSAMPLE)g; out[2] = (JSAMPLE)r; out[3] = 255; break;
        case 1: out[0] = (JSAMPLE)r; out[1] = (JSAMPLE)g; out[2] = (JSAMPLE)b; out[3] = 255; break;
        case 2: out[0] = 255; out[1] = (JSAMPLE)b; out[2] = (JSAMPLE)g; out[3] = (JSAMPLE)r; break;
        default: out[0] = 255; out[1] = (JSAMPLE)r; out[2] = (JSAMPLE)g; out[3] = (JSAMPLE)b; break;
        }
        out += 4;
      }
    }
  }
}
/* ==== END fn=jsimd_ycc_rgb_convert_lsx ==== */
