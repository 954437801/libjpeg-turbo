/*
 * lsxint.h
 *
 * Shared helpers for the LoongArch LSX SIMD kernels: unsigned -> signed
 * widening, saturating pack to 8-bit, and pixel interleaving.
 *
 * Numerical semantics follow the C reference implementation byte for byte
 * (16.16 fixed point + saturation, equivalent to libjpeg's range_limit
 * table lookup).
 *
 * Copyright (C) 2026, Loongson Technology Corporation Limited.
 */

#ifndef LSXINT_H
#define LSXINT_H

#include "../../jinclude.h"
#include "../../jpeglib.h"

#include <lsxintrin.h>
#include <string.h>

/* libjpeg defines INLINE in jconfigint.h; keep this header usable on its own */
#ifndef INLINE
#define INLINE inline
#endif

#define L_FIX(x)   ((int)((x) * 65536.0 + 0.5))
#define L_ONE_HALF (1 << 15)
#define L_CENTER   128

/* ==== BEGIN fn=lsx_load8_u16 ==== */
/* Load 8 u8 and zero-extend into v8u16 (vilvl_b places its second operand
 * first, so `raw` must come second to keep the byte order) */
static INLINE __m128i
lsx_load8_u16(const JSAMPLE *p)
{
  __m128i raw = __lsx_vld(p, 0);
  return __lsx_vilvl_b(__lsx_vreplgr2vr_b(0), raw);
}
/* ==== END fn=lsx_load8_u16 ==== */

/* ==== BEGIN fn=lsx_sext ==== */
/* 16 -> 32 bit sign extension: vilvl_h only concatenates bits, so the sign
 * mask has to be manufactured with vsrai_h */
static INLINE __m128i
lsx_sext_lo(__m128i v16)
{
  return __lsx_vilvl_h(__lsx_vsrai_h(v16, 15), v16);
}

static INLINE __m128i
lsx_sext_hi(__m128i v16)
{
  return __lsx_vilvh_h(__lsx_vsrai_h(v16, 15), v16);
}
/* ==== END fn=lsx_sext ==== */

/* ==== BEGIN fn=lsx_pack8 ==== */
/* Two v4i32 -> 8 values saturated to 0..255 (low 8 bytes valid).
 * vssrarni_h_w rounds; vssrarni_bu_h saturates unsigned. */
static INLINE __m128i
lsx_pack8(__m128i lo, __m128i hi)
{
  __m128i v16 = __lsx_vssrarni_h_w(hi, lo, 0);              /* low 4 from lo */
  return __lsx_vssrarni_bu_h(__lsx_vreplgr2vr_b(0), v16, 0);
}
/* ==== END fn=lsx_pack8 ==== */

/* ==== BEGIN fn=lsx_interleave4 ==== */
/* Interleave 8 pixels of R/G/B/A into two 128-bit words.
 * mode: 0=BGRX 1=RGBX 2=XBGR 3=XRGB
 *
 * Element order was measured with ixprobe.c, not assumed:
 *   vilvl_b(x, y) = [y0, x0, y1, x1, ..., y7, x7]   (second operand first)
 *   vilvl_h(x, y) = 16-bit interleave of bytes 0..7, y first
 *   vilvh_h(x, y) = 16-bit interleave of bytes 8..15, y first
 * Note the component vectors only carry 8 valid bytes (pixels 0..7), so the
 * high half must be taken from the *pairs*, never with vilvh_b.
 * So a 4-byte pixel stream is built as:
 *   pair1 = vilvl_b(src_c1, src_c0)  ->  c0 c1 c0 c1 ...  (all 8 pixels)
 *   pair2 = vilvl_b(src_c3, src_c2)  ->  c2 c3 c2 c3 ...
 *   lo    = vilvl_h(pair2, pair1)    ->  pixels 0..3
 *   hi    = vilvh_h(pair2, pair1)    ->  pixels 4..7 */
static INLINE void
lsx_interleave4(__m128i r8, __m128i g8, __m128i b8, int mode,
                __m128i *lo, __m128i *hi)
{
  const __m128i ff = __lsx_vreplgr2vr_b((char)0xff);
  __m128i p1, p2;

  switch (mode) {
  case 0: /* B G R X */
    p1 = __lsx_vilvl_b(g8, b8);  p2 = __lsx_vilvl_b(ff, r8);
    break;
  case 1: /* R G B X */
    p1 = __lsx_vilvl_b(g8, r8);  p2 = __lsx_vilvl_b(ff, b8);
    break;
  case 2: /* X B G R */
    p1 = __lsx_vilvl_b(b8, ff);  p2 = __lsx_vilvl_b(r8, g8);
    break;
  default: /* X R G B */
    p1 = __lsx_vilvl_b(r8, ff);  p2 = __lsx_vilvl_b(b8, g8);
    break;
  }

  *lo = __lsx_vilvl_h(p2, p1);   /* pixels 0..3 */
  *hi = __lsx_vilvh_h(p2, p1);   /* pixels 4..7 */
}
/* ==== END fn=lsx_interleave4 ==== */

/* ==== BEGIN fn=lsx_store3 ==== */
/* Store 8 pixels as 3 bytes each by dropping the pad byte from the 32-byte
 * BGRX/RGBX interleaving.
 * vshuf_b(a, b, c): indices 0..15 select from b, 16..31 from a (verified
 * with shufprobe.c) */
static const unsigned char k_idx3_lo[16] = { 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20 };
static const unsigned char k_idx3_hi[16] = { 21, 22, 24, 25, 26, 28, 29, 30, 0, 0, 0, 0, 0, 0, 0, 0 };

static INLINE void
lsx_store3(__m128i lo, __m128i hi, JSAMPLE *out)
{
  const __m128i il = __lsx_vld(k_idx3_lo, 0);
  const __m128i ih = __lsx_vld(k_idx3_hi, 0);
  __m128i o0 = __lsx_vshuf_b(hi, lo, il);
  __m128i o1 = __lsx_vshuf_b(hi, lo, ih);
  union { __m128i v; JSAMPLE b[16]; } u;
  __lsx_vst(o0, out, 0);
  u.v = o1;
  memcpy(out + 16, u.b, 8);
}
/* ==== END fn=lsx_store3 ==== */

#endif /* LSXINT_H */
