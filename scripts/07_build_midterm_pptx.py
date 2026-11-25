"""Build the 14-slide PKU minimalist academic Midterm Presentation (.pptx)
strictly following docs/ppt_design_spec.md (inspired by hugohe3/ppt-master's
Spec-First absolute-coordinate DrawingML pipeline).

Produces:
  1. D:\\QuantProjects\\tscs-alpha\\docs\\TS-CSNet_中期汇报PPT.pptx
     (100% natively editable PowerPoint / WPS OOXML shapes, tables, text & charts)
  2. D:\\QuantProjects\\tscs-alpha\\figures\\ppt_previews\\slide_01.png .. slide_14.png
     (1:1 1280x720 visual preview renders for strict layout & typography audit)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile

from lxml import etree
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(r"D:\QuantProjects\tscs-alpha")
FIG_DIR = PROJECT_ROOT / "figures"
OUT_PPTX = PROJECT_ROOT / "docs" / "中证500量价与基本面融合增强策略_中期汇报PPT.pptx"
PREVIEW_DIR = FIG_DIR / "ppt_previews"

# Logical 16:9 canvas: 1280 x 720 px -> 12192000 x 6858000 EMU (1 px = 9525 EMU)
W_PX, H_PX = 1280, 720
EMU = 9525
W_EMU, H_EMU = W_PX * EMU, H_PX * EMU

# PKU Academic Minimalist Palette
C_CRIMSON = "8B1A24"
C_NAVY = "1B365D"
C_STEEL = "2B5B84"
C_GREEN = "1B6B4F"
C_RED = "B91C1C"
C_INK = "1E293B"
C_MUTED = "475569"
C_LIGHT_TXT = "64748B"
C_WHITE = "FFFFFF"
C_BG_CARD = "F5F7FA"
C_BG_CRIMSON = "FDF5F6"
C_BG_GREEN = "F2F9F6"
C_BG_NAVY_SOFT = "EFF4FA"
C_BORDER = "CBD5E1"
C_RULE = "E2E8F0"

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"


def _hex_rgb(h: str) -> Tuple[int, int, int]:
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _pil_font(pt: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    # 1 pt on 96-DPI 1280x720 (13.333in) slide = 1.333 px
    px = max(10, int(round(pt * 1.333)))
    path = FONT_BOLD if bold else FONT_REG
    return ImageFont.truetype(path, px)


def _wrap_text_pil(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
    lines: List[str] = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue
        cur = ""
        for ch in raw_line:
            test = cur + ch
            if draw.textlength(test, font=font) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


@dataclass
class SlideSpec:
    slide_no: int
    total_slides: int
    bg_color: str = C_WHITE
    shapes_xml: List[etree._Element] = field(default_factory=list)
    images: List[Tuple[str, Path]] = field(default_factory=list)  # (rId, path)
    _shape_id: int = 2
    _img: Image.Image = field(init=False)
    _draw: ImageDraw.ImageDraw = field(init=False)

    def __post_init__(self) -> None:
        self._img = Image.new("RGB", (W_PX, H_PX), _hex_rgb(self.bg_color))
        self._draw = ImageDraw.Draw(self._img)

    def _next_id(self) -> int:
        sid = self._shape_id
        self._shape_id += 1
        return sid

    def add_rect(
        self,
        x: float, y: float, w: float, h: float,
        fill: Optional[str] = None,
        border: Optional[str] = None,
        border_pt: float = 0.75,
        rounded: bool = False,
        radius_px: int = 8,
    ) -> None:
        sid = self._next_id()
        sp = etree.Element(f"{{{NS_P}}}sp")
        nv = etree.SubElement(sp, f"{{{NS_P}}}nvSpPr")
        etree.SubElement(nv, f"{{{NS_P}}}cNvPr", id=str(sid), name=f"Rect_{sid}")
        etree.SubElement(nv, f"{{{NS_P}}}cNvSpPr")
        etree.SubElement(nv, f"{{{NS_P}}}nvPr")

        spPr = etree.SubElement(sp, f"{{{NS_P}}}spPr")
        xfrm = etree.SubElement(spPr, f"{{{NS_A}}}xfrm")
        etree.SubElement(xfrm, f"{{{NS_A}}}off", x=str(int(x * EMU)), y=str(int(y * EMU)))
        etree.SubElement(xfrm, f"{{{NS_A}}}ext", cx=str(int(w * EMU)), cy=str(int(h * EMU)))
        prst = etree.SubElement(spPr, f"{{{NS_A}}}prstGeom", prst="roundRect" if rounded else "rect")
        avLst = etree.SubElement(prst, f"{{{NS_A}}}avLst")
        if rounded:
            adj_val = int(min(25000, max(3000, (radius_px / max(1.0, min(w, h))) * 100000)))
            etree.SubElement(avLst, f"{{{NS_A}}}gd", name="adj", fmla=f"val {adj_val}")

        if fill:
            sf = etree.SubElement(spPr, f"{{{NS_A}}}solidFill")
            etree.SubElement(sf, f"{{{NS_A}}}srgbClr", val=fill)
        else:
            etree.SubElement(spPr, f"{{{NS_A}}}noFill")

        if border:
            ln = etree.SubElement(spPr, f"{{{NS_A}}}ln", w=str(int(border_pt * 12700)))
            sf_b = etree.SubElement(ln, f"{{{NS_A}}}solidFill")
            etree.SubElement(sf_b, f"{{{NS_A}}}srgbClr", val=border)
        else:
            ln = etree.SubElement(spPr, f"{{{NS_A}}}ln")
            etree.SubElement(ln, f"{{{NS_A}}}noFill")

        self.shapes_xml.append(sp)

        # PIL preview
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        f_rgb = _hex_rgb(fill) if fill else None
        b_rgb = _hex_rgb(border) if border else None
        bw = max(1, int(round(border_pt * 1.2))) if border else 0
        if rounded:
            self._draw.rounded_rectangle(bbox, radius=radius_px, fill=f_rgb, outline=b_rgb, width=bw)
        else:
            self._draw.rectangle(bbox, fill=f_rgb, outline=b_rgb, width=bw)

    def add_text(
        self,
        x: float, y: float, w: float, h: float,
        text: str,
        size: float = 12.0,
        color: str = C_INK,
        bold: bool = False,
        align: str = "left",      # left | center | right
        valign: str = "top",      # top | middle | bottom
        line_spacing: float = 1.25,
        pad_x: float = 4.0,
        pad_y: float = 2.0,
    ) -> None:
        sid = self._next_id()
        sp = etree.Element(f"{{{NS_P}}}sp")
        nv = etree.SubElement(sp, f"{{{NS_P}}}nvSpPr")
        etree.SubElement(nv, f"{{{NS_P}}}cNvPr", id=str(sid), name=f"Text_{sid}")
        etree.SubElement(nv, f"{{{NS_P}}}cNvSpPr", txBox="1")
        etree.SubElement(nv, f"{{{NS_P}}}nvPr")

        spPr = etree.SubElement(sp, f"{{{NS_P}}}spPr")
        xfrm = etree.SubElement(spPr, f"{{{NS_A}}}xfrm")
        etree.SubElement(xfrm, f"{{{NS_A}}}off", x=str(int(x * EMU)), y=str(int(y * EMU)))
        etree.SubElement(xfrm, f"{{{NS_A}}}ext", cx=str(int(w * EMU)), cy=str(int(h * EMU)))
        prst = etree.SubElement(spPr, f"{{{NS_A}}}prstGeom", prst="rect")
        etree.SubElement(prst, f"{{{NS_A}}}avLst")
        etree.SubElement(spPr, f"{{{NS_A}}}noFill")

        txBody = etree.SubElement(sp, f"{{{NS_P}}}txBody")
        anchor_map = {"top": "t", "middle": "ctr", "bottom": "b"}
        etree.SubElement(
            txBody, f"{{{NS_A}}}bodyPr",
            wrap="square",
            anchor=anchor_map.get(valign, "t"),
            lIns=str(int(pad_x * EMU)),
            rIns=str(int(pad_x * EMU)),
            tIns=str(int(pad_y * EMU)),
            bIns=str(int(pad_y * EMU)),
        )
        etree.SubElement(txBody, f"{{{NS_A}}}lstStyle")

        algn_map = {"left": "l", "center": "ctr", "right": "r"}
        for para_txt in text.split("\n"):
            p = etree.SubElement(txBody, f"{{{NS_A}}}p")
            pPr = etree.SubElement(p, f"{{{NS_A}}}pPr", algn=algn_map.get(align, "l"))
            lnSpc = etree.SubElement(pPr, f"{{{NS_A}}}lnSpc")
            etree.SubElement(lnSpc, f"{{{NS_A}}}spcPct", val=str(int(line_spacing * 100000)))
            r = etree.SubElement(p, f"{{{NS_A}}}r")
            rPr = etree.SubElement(
                r, f"{{{NS_A}}}rPr",
                lang="zh-CN",
                sz=str(int(size * 100)),
                b="1" if bold else "0",
                dirty="0",
            )
            sf = etree.SubElement(rPr, f"{{{NS_A}}}solidFill")
            etree.SubElement(sf, f"{{{NS_A}}}srgbClr", val=color)
            etree.SubElement(rPr, f"{{{NS_A}}}latin", typeface="Microsoft YaHei")
            etree.SubElement(rPr, f"{{{NS_A}}}ea", typeface="Microsoft YaHei")
            t_el = etree.SubElement(r, f"{{{NS_A}}}t")
            t_el.text = para_txt

        self.shapes_xml.append(sp)

        # PIL preview
        font = _pil_font(size, bold=bold)
        inner_w = max(20, int(w - 2 * pad_x))
        lines = _wrap_text_pil(self._draw, text, font, inner_w)
        line_h = int(round(size * 1.333 * line_spacing))
        total_h = len(lines) * line_h
        if valign == "middle":
            cur_y = y + max(pad_y, (h - total_h) / 2.0)
        elif valign == "bottom":
            cur_y = y + max(pad_y, h - total_h - pad_y)
        else:
            cur_y = y + pad_y

        rgb = _hex_rgb(color)
        for ln in lines:
            lw = self._draw.textlength(ln, font=font)
            if align == "center":
                cur_x = x + (w - lw) / 2.0
            elif align == "right":
                cur_x = x + w - pad_x - lw
            else:
                cur_x = x + pad_x
            self._draw.text((int(cur_x), int(cur_y)), ln, font=font, fill=rgb)
            cur_y += line_h

    def add_image(self, x: float, y: float, max_w: float, max_h: float, img_path: Path, border: Optional[str] = None) -> None:
        """Fit image cleanly inside (x, y, max_w, max_h) preserving exact aspect ratio."""
        with Image.open(img_path) as im:
            iw, ih = im.size
        scale = min(max_w / iw, max_h / ih)
        fw, fh = iw * scale, ih * scale
        fx = x + (max_w - fw) / 2.0
        fy = y + (max_h - fh) / 2.0

        if border:
            self.add_rect(fx - 2, fy - 2, fw + 4, fh + 4, fill=C_WHITE, border=border, border_pt=0.75, rounded=False)

        rid = f"rIdImg{len(self.images) + 1}"
        self.images.append((rid, img_path))

        sid = self._next_id()
        pic = etree.Element(f"{{{NS_P}}}pic")
        nv = etree.SubElement(pic, f"{{{NS_P}}}nvPicPr")
        etree.SubElement(nv, f"{{{NS_P}}}cNvPr", id=str(sid), name=f"Pic_{sid}")
        cNvPic = etree.SubElement(nv, f"{{{NS_P}}}cNvPicPr")
        etree.SubElement(cNvPic, f"{{{NS_A}}}picLocks", noChangeAspect="1")
        etree.SubElement(nv, f"{{{NS_P}}}nvPr")

        blipFill = etree.SubElement(pic, f"{{{NS_P}}}blipFill")
        etree.SubElement(blipFill, f"{{{NS_A}}}blip", {f"{{{NS_R}}}embed": rid})
        stretch = etree.SubElement(blipFill, f"{{{NS_A}}}stretch")
        etree.SubElement(stretch, f"{{{NS_A}}}fillRect")

        spPr = etree.SubElement(pic, f"{{{NS_P}}}spPr")
        xfrm = etree.SubElement(spPr, f"{{{NS_A}}}xfrm")
        etree.SubElement(xfrm, f"{{{NS_A}}}off", x=str(int(fx * EMU)), y=str(int(fy * EMU)))
        etree.SubElement(xfrm, f"{{{NS_A}}}ext", cx=str(int(fw * EMU)), cy=str(int(fh * EMU)))
        prst = etree.SubElement(spPr, f"{{{NS_A}}}prstGeom", prst="rect")
        etree.SubElement(prst, f"{{{NS_A}}}avLst")

        self.shapes_xml.append(pic)

        # PIL preview
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB").resize((max(1, int(round(fw))), max(1, int(round(fh)))), Image.Resampling.LANCZOS)
            self._img.paste(im_rgb, (int(round(fx)), int(round(fy))))

    def add_table(
        self,
        x: float, y: float, w: float,
        col_widths: List[float],
        headers: List[str],
        rows: List[List[str]],
        row_h: float = 34.0,
        header_h: float = 36.0,
        font_size: float = 9.8,
        header_bg: str = C_NAVY,
        highlight_rows: Optional[dict] = None,  # row_idx -> fill_hex
        col_aligns: Optional[List[str]] = None,
    ) -> float:
        """Emit a native PowerPoint <a:tbl> table + exact PIL preview. Returns total height."""
        highlight_rows = highlight_rows or {}
        col_aligns = col_aligns or ["left"] * len(headers)
        tot_h = header_h + row_h * len(rows)

        sid = self._next_id()
        gf = etree.Element(f"{{{NS_P}}}graphicFrame")
        nv = etree.SubElement(gf, f"{{{NS_P}}}nvGraphicFramePr")
        etree.SubElement(nv, f"{{{NS_P}}}cNvPr", id=str(sid), name=f"Table_{sid}")
        cNvGf = etree.SubElement(nv, f"{{{NS_P}}}cNvGraphicFramePr")
        etree.SubElement(cNvGf, f"{{{NS_A}}}graphicFrameLocks", noGrp="1")
        etree.SubElement(nv, f"{{{NS_P}}}nvPr")

        xfrm = etree.SubElement(gf, f"{{{NS_P}}}xfrm")
        etree.SubElement(xfrm, f"{{{NS_A}}}off", x=str(int(x * EMU)), y=str(int(y * EMU)))
        etree.SubElement(xfrm, f"{{{NS_A}}}ext", cx=str(int(w * EMU)), cy=str(int(tot_h * EMU)))

        graphic = etree.SubElement(gf, f"{{{NS_A}}}graphic")
        gdata = etree.SubElement(graphic, f"{{{NS_A}}}graphicData", uri="http://schemas.openxmlformats.org/drawingml/2006/table")
        tbl = etree.SubElement(gdata, f"{{{NS_A}}}tbl")
        etree.SubElement(tbl, f"{{{NS_A}}}tblPr", firstRow="1", bandRow="0")
        tblGrid = etree.SubElement(tbl, f"{{{NS_A}}}tblGrid")
        for cw in col_widths:
            etree.SubElement(tblGrid, f"{{{NS_A}}}gridCol", w=str(int(cw * EMU)))

        algn_map = {"left": "l", "center": "ctr", "right": "r"}

        def _add_row_xml(cells: List[str], rh: float, is_hdr: bool, r_idx: int) -> None:
            tr = etree.SubElement(tbl, f"{{{NS_A}}}tr", h=str(int(rh * EMU)))
            bg = header_bg if is_hdr else highlight_rows.get(r_idx, C_WHITE if r_idx % 2 == 0 else C_BG_CARD)
            fg = C_WHITE if is_hdr else C_INK
            for ci, val in enumerate(cells):
                tc = etree.SubElement(tr, f"{{{NS_A}}}tc")
                txBody = etree.SubElement(tc, f"{{{NS_A}}}txBody")
                etree.SubElement(txBody, f"{{{NS_A}}}bodyPr", wrap="square", anchor="ctr")
                etree.SubElement(txBody, f"{{{NS_A}}}lstStyle")
                p = etree.SubElement(txBody, f"{{{NS_A}}}p")
                etree.SubElement(p, f"{{{NS_A}}}pPr", algn=algn_map.get(col_aligns[ci], "l"))
                r = etree.SubElement(p, f"{{{NS_A}}}r")
                cell_bold = is_hdr or (ci == 0)
                cell_fg = fg
                if not is_hdr:
                    if val.startswith("+") and "%" in val:
                        cell_fg = C_GREEN
                        cell_bold = True
                    elif val.startswith("-") and "%" in val:
                        cell_fg = C_RED
                rPr = etree.SubElement(r, f"{{{NS_A}}}rPr", lang="zh-CN", sz=str(int(font_size * 100)), b="1" if cell_bold else "0")
                sf = etree.SubElement(rPr, f"{{{NS_A}}}solidFill")
                etree.SubElement(sf, f"{{{NS_A}}}srgbClr", val=cell_fg)
                etree.SubElement(rPr, f"{{{NS_A}}}latin", typeface="Microsoft YaHei")
                etree.SubElement(rPr, f"{{{NS_A}}}ea", typeface="Microsoft YaHei")
                t_el = etree.SubElement(r, f"{{{NS_A}}}t")
                t_el.text = val

                tcPr = etree.SubElement(tc, f"{{{NS_A}}}tcPr", anchor="ctr", marL=str(int(6 * EMU)), marR=str(int(6 * EMU)), marT=str(int(3 * EMU)), marB=str(int(3 * EMU)))
                for b_tag in ("lnB",):
                    ln = etree.SubElement(tcPr, f"{{{NS_A}}}{b_tag}", w="9525")
                    sfb = etree.SubElement(ln, f"{{{NS_A}}}solidFill")
                    etree.SubElement(sfb, f"{{{NS_A}}}srgbClr", val=C_BORDER)
                sfill = etree.SubElement(tcPr, f"{{{NS_A}}}solidFill")
                etree.SubElement(sfill, f"{{{NS_A}}}srgbClr", val=bg)

        _add_row_xml(headers, header_h, True, -1)
        for ri, row in enumerate(rows):
            _add_row_xml(row, row_h, False, ri)

        self.shapes_xml.append(gf)

        # PIL preview for table
        cur_y = y
        all_rows = [(headers, header_h, True, -1)] + [(r, row_h, False, idx) for idx, r in enumerate(rows)]
        for cells, rh, is_hdr, ri in all_rows:
            bg = header_bg if is_hdr else highlight_rows.get(ri, C_WHITE if ri % 2 == 0 else C_BG_CARD)
            cur_x = x
            for ci, val in enumerate(cells):
                cw = col_widths[ci]
                self._draw.rectangle([int(cur_x), int(cur_y), int(cur_x + cw), int(cur_y + rh)], fill=_hex_rgb(bg))
                self._draw.line([(int(cur_x), int(cur_y + rh)), (int(cur_x + cw), int(cur_y + rh))], fill=_hex_rgb(C_BORDER), width=1)
                cell_bold = is_hdr or (ci == 0)
                cell_fg = C_WHITE if is_hdr else C_INK
                if not is_hdr:
                    if val.startswith("+") and "%" in val:
                        cell_fg = C_GREEN
                        cell_bold = True
                    elif val.startswith("-") and "%" in val:
                        cell_fg = C_RED
                font = _pil_font(font_size, bold=cell_bold)
                lines = _wrap_text_pil(self._draw, val, font, max(20, int(cw - 12)))
                lh = int(round(font_size * 1.333 * 1.15))
                ty = cur_y + max(2, (rh - len(lines) * lh) / 2.0)
                for ln in lines:
                    lw = self._draw.textlength(ln, font=font)
                    if col_aligns[ci] == "center":
                        tx = cur_x + (cw - lw) / 2.0
                    elif col_aligns[ci] == "right":
                        tx = cur_x + cw - 6 - lw
                    else:
                        tx = cur_x + 6
                    self._draw.text((int(tx), int(ty)), ln, font=font, fill=_hex_rgb(cell_fg))
                    ty += lh
                cur_x += cw
            cur_y += rh

        return tot_h

    def add_header_footer(self, kicker: str, title: str) -> None:
        # Top thin crimson bar across full slide
        self.add_rect(0, 0, W_PX, 5, fill=C_CRIMSON)
        # Kicker & main headline
        self.add_text(56, 16, 680, 22, kicker, size=10.5, color=C_CRIMSON, bold=True)
        self.add_text(56, 38, 1168, 40, title, size=20.5, color=C_NAVY, bold=True)
        # Two-tone academic rule
        self.add_rect(56, 84, 140, 2.5, fill=C_CRIMSON)
        self.add_rect(196, 85, 1028, 1.0, fill=C_RULE)
        # Footer
        self.add_rect(56, 684, 1168, 1.0, fill=C_RULE)
        self.add_text(56, 689, 800, 22, "北京大学量化投资课程项目中期汇报  |  中证500量价与基本面融合增强策略", size=8.8, color=C_LIGHT_TXT)
        self.add_text(1080, 689, 144, 22, f"{self.slide_no:02d} / {self.total_slides:02d}", size=9.0, color=C_NAVY, bold=True, align="right")

    def add_kpi_card(
        self, x: float, y: float, w: float, h: float,
        label: str, value: str, sub: str,
        accent: str = C_NAVY, bg: str = C_BG_CARD,
    ) -> None:
        self.add_rect(x, y, w, h, fill=bg, border=C_BORDER, border_pt=0.75, rounded=True, radius_px=8)
        self.add_rect(x, y, 5, h, fill=accent)
        self.add_text(x + 14, y + 10, w - 20, 22, label, size=10.0, color=C_MUTED, bold=True)
        self.add_text(x + 14, y + 34, w - 20, 36, value, size=21.0, color=accent, bold=True)
        self.add_text(x + 14, y + 73, w - 20, h - 78, sub, size=9.3, color=C_LIGHT_TXT)


def build_all_slides() -> List[SlideSpec]:
    total = 14
    slides: List[SlideSpec] = []

    # =========================================================================
    # SLIDE 1: COVER (Minimalist PKU Academic Cover)
    # =========================================================================
    s = SlideSpec(1, total, bg_color=C_WHITE)
    s.add_rect(0, 0, 28, H_PX, fill=C_CRIMSON)
    s.add_rect(28, 0, 8, H_PX, fill=C_NAVY)
    s.add_rect(0, H_PX - 66, W_PX, 66, fill=C_BG_CARD)
    s.add_rect(36, H_PX - 66, W_PX - 36, 1.5, fill=C_RULE)

    s.add_rect(96, 108, 252, 32, fill=C_BG_CRIMSON, border=C_CRIMSON, border_pt=1.0, rounded=True, radius_px=6)
    s.add_text(96, 110, 252, 28, "北京大学 · 量化投资课程项目中期汇报", size=10.5, color=C_CRIMSON, bold=True, align="center", valign="middle")

    s.add_text(96, 170, 1080, 92, "中证500量价与基本面融合增强策略", size=34.0, color=C_NAVY, bold=True)
    s.add_text(96, 254, 1060, 44, "量化投资课程项目中期进展汇报：投资逻辑、实证归因与模拟建仓", size=17.0, color=C_MUTED)
    s.add_rect(96, 316, 180, 4, fill=C_CRIMSON)

    # 4 positioning highlight pills on cover
    cover_items = [
        ("产品定位", "机构指增模拟基金", "1,000 万初始模拟资金"),
        ("投资范围", "中证 500 成分股", "周度 PIT 快照防幸存者偏差"),
        ("核心逻辑", "量价融合基本面", "多因子相对排序 + 软倾斜配置"),
        ("研究进度", "9年回测 + 实盘建仓", "2026-11-24 建仓计划冻结"),
    ]
    for i, (k, v, sub) in enumerate(cover_items):
        cx = 96 + i * 276
        s.add_rect(cx, 368, 256, 126, fill=C_BG_CARD, border=C_BORDER, border_pt=0.8, rounded=True, radius_px=8)
        s.add_rect(cx, 368, 256, 5, fill=C_NAVY if i % 2 == 0 else C_CRIMSON)
        s.add_text(cx + 16, 386, 224, 22, k, size=10.0, color=C_LIGHT_TXT, bold=True)
        s.add_text(cx + 16, 412, 224, 34, v, size=15.5, color=C_NAVY, bold=True)
        s.add_text(cx + 16, 452, 224, 28, sub, size=9.8, color=C_MUTED)

    s.add_text(96, 540, 1060, 56, "汇报主线：投资思想与产品定位 (15%)  →  选股/择时/配置/风控流程 (25%)  →  回测/持仓/归因/适用性分析 (40%)  →  模拟交易配置 (20%)", size=11.0, color=C_STEEL, bold=True)
    s.add_text(96, H_PX - 46, 650, 28, "汇报人 / 小组成员：________________      业绩比较基准：中证 500 全收益指数 (H00905)", size=10.5, color=C_MUTED)
    s.add_text(920, H_PX - 46, 290, 28, "汇报日期：2026 年 11 月", size=10.5, color=C_NAVY, bold=True, align="right")
    slides.append(s)

    # =========================================================================
    # SLIDE 2: PART 1 - 策略背景、基金定位与中证500投资逻辑 (15%)
    # =========================================================================
    s = SlideSpec(2, total)
    s.add_header_footer("PART 01  |  策略背景与基本投资思想 (15%)", "定位 1,000 万中证 500 指数增强基金：以“头部超配 + 尾部排雷”追求长期稳健增值")

    s.add_kpi_card(56, 106, 276, 104, "产品定位与资金规模", "1,000 万元", "机构/公募指增定位 · 长期稳健增值", C_NAVY)
    s.add_kpi_card(352, 106, 276, 104, "投资范围", "中证 500 成分股", "周度 PIT 成分快照 · 500 只细分龙头", C_STEEL)
    s.add_kpi_card(648, 106, 276, 104, "业绩比较基准", "H00905 全收益", "含分红再投资 · 严守高仓位跟踪", C_CRIMSON)
    s.add_kpi_card(944, 106, 280, 104, "超额收益来源", "超配 + 排雷", "相对强弱排序 · 双轮驱动低换手", C_GREEN)

    # Left panel: Why CSI 500?
    s.add_rect(56, 234, 568, 424, fill=C_BG_CARD, border=C_BORDER, rounded=True)
    s.add_rect(56, 234, 568, 42, fill=C_NAVY, rounded=False)
    s.add_text(74, 242, 530, 28, "一、为什么选择中证 500 指数增强赛道？", size=13.0, color=C_WHITE, bold=True)

    reasons = [
        ("1. 定价效率适中，横截面选股 Alpha 土壤肥沃",
         "相较沪深 300 机构高度定价、个股分化有限，中证 500 覆盖电子、通信、医药、电力设备等中盘成长龙头，横截面收益离散度高。"),
        ("2. 流动性均衡，兼顾容量与抗极端风险能力",
         "相较中证 1000 与微盘股在流动性收紧时易发生剧烈踩踏（如 2024 年 2 月微盘危机），中证 500 流动性更深，支撑千万至亿元级调仓。"),
        ("3. 严格对标含分红全收益基准，避免牛市踏空",
         "不做激进大盘择时，保持约 99% 高权益仓位紧跟基准，把研究重心聚焦于确定性更高的横截面相对强弱选股。"),
    ]
    cy = 294
    for rt, rb in reasons:
        s.add_text(74, cy, 532, 26, rt, size=11.8, color=C_NAVY, bold=True)
        s.add_text(74, cy + 28, 532, 74, rb, size=10.6, color=C_MUTED, line_spacing=1.32)
        cy += 116

    # Right panel: Dual-Engine Alpha Logic
    s.add_rect(648, 234, 576, 424, fill=C_WHITE, border=C_BORDER, rounded=True)
    s.add_rect(648, 234, 576, 42, fill=C_CRIMSON, rounded=False)
    s.add_text(666, 242, 540, 28, "二、纯多头指数增强的“双引擎”超额逻辑", size=13.0, color=C_WHITE, bold=True)

    s.add_rect(672, 298, 528, 128, fill=C_BG_GREEN, border=C_GREEN, border_pt=1.0, rounded=True)
    s.add_text(688, 308, 496, 26, "引擎 A：头部优质标的适度超配（精选赢家）", size=12.0, color=C_GREEN, bold=True)
    s.add_text(688, 338, 496, 78, "在流通市值基准权重之上，对估值安全、盈利改善且短期量价未过热的头部高分股给予最高 2.5× 权重倾斜，稳步捕获行业内领跑股超额收益。", size=10.6, color=C_INK, line_spacing=1.32)

    s.add_rect(672, 444, 528, 128, fill=C_BG_CRIMSON, border=C_CRIMSON, border_pt=1.0, rounded=True)
    s.add_text(688, 454, 496, 26, "引擎 B：尾部弱势标的低配与清零（精准排雷）", size=12.0, color=C_CRIMSON, bold=True)
    s.add_text(688, 484, 496, 78, "A 股纯多头组合无法低成本做空，但将高估值、亏损恶化、短期暴涨高换手的尾部约 80 只风险股权重直接清零，等价于相对基准建立了精准空头保护。", size=10.6, color=C_INK, line_spacing=1.32)

    s.add_text(672, 594, 528, 48, "实证预告：后文归因分析证明，组合毛超额中约 36% 来自头部超配，约 64% 来自低配/清零尾部弱势股！", size=10.5, color=C_NAVY, bold=True)
    slides.append(s)

    # =========================================================================
    # SLIDE 3: PART 1 - 因子金融逻辑与引入机器学习的价值 (15%)
    # =========================================================================
    s = SlideSpec(3, total)
    s.add_header_footer("PART 01  |  策略背景与基本投资思想 (15%)", "严选纯 PIT 量价与基本面因子，借力机器学习捕捉非线性交互与走势形态")

    # Top 3 Factor Pillar Cards
    pillars = [
        ("短期量价行为（20日×6通道）", C_NAVY, C_BG_NAVY_SOFT,
         "包含：日收益、隔夜跳空、日内振幅、收盘相对位置、对数换手率、相对20日均量比。",
         "金融逻辑：A 股散户交易占比高，短期急涨伴随异常高换手往往意味筹码拥挤与过度反应；量价平稳、温和回调的标的更具均值回归修复动力。"),
        ("估值安全边际（纯PIT日频）", C_CRIMSON, C_BG_CRIMSON,
         "包含：盈利收益率 EP_TTM (1/PE)、账面市值比 BP_MRQ (1/PB)、对数流通市值。",
         "金融逻辑：直接采用公告日实时更新的估值指标（彻底抛弃有重述漏洞的免费季报），衡量单位市值的盈利与净资产安全垫。"),
        ("盈利质量与波动控制", C_GREEN, C_BG_GREEN,
         "包含：估值反推隐含 ROE (PB/PE)、60日 ΔROE、60日动量与波动率、20日换手变异系数。",
         "金融逻辑：联合识别“真低估高景气”与“持续亏损的价值陷阱”，同时刻画换手稳定性（量稳换手率）以规避短期投机噪音。"),
    ]
    for i, (pt, pc, pbg, p1, p2) in enumerate(pillars):
        px = 56 + i * 396
        s.add_rect(px, 108, 376, 242, fill=pbg, border=C_BORDER, rounded=True)
        s.add_rect(px, 108, 376, 38, fill=pc)
        s.add_text(px + 14, 115, 348, 24, pt, size=11.5, color=C_WHITE, bold=True)
        s.add_text(px + 14, 156, 348, 66, p1, size=10.2, color=C_INK, bold=True, line_spacing=1.28)
        s.add_text(px + 14, 224, 348, 116, p2, size=10.0, color=C_MUTED, line_spacing=1.32)

    # Bottom comparison table: Traditional Multi-Factor vs Lightweight ML
    s.add_text(56, 372, 600, 28, "为什么引入轻量级机器学习？（相较传统线性多因子的三大增量价值）", size=12.8, color=C_NAVY, bold=True)
    s.add_table(
        x=56, y=406, w=1168,
        col_widths=[180, 320, 360, 308],
        headers=["对比维度", "传统线性多因子（如岭回归/等权打分）", "本项目机器学习思路（TS-CSNet）", "带来的实际投资价值"],
        rows=[
            ["因子间交互关系", "假设各因子线性叠加、彼此独立", "自动刻画“低估+盈利改善+缩量回调”非线性交互", "有效避开低估值亏损陷阱与放量滞涨"],
            ["短期走势形态", "仅看单一窗口均值，忽略涨跌先后节奏", "直接读取 20 日完整量价序列形态", "减少人工堆砌技术指标与过拟合风险"],
            ["横截面相对比较", "单只股票孤立打分，易受大盘波动干扰", "500 只成分股同日横向交互 + 无偏排序目标", "样本外 RankIC 由 +4.95% 升至 +7.63%"],
            ["数据与复杂度纪律", "依赖静态财务表，难适应风格切换", "仅 1.5 万轻量参数 + 年度滚动 + 门控自适应", "普通 CPU 可复现，逻辑透明可解释"],
        ],
        row_h=54.0, header_h=36.0, font_size=9.8,
        col_aligns=["left", "left", "left", "left"],
    )
    slides.append(s)

    # =========================================================================
    # SLIDE 4: PART 2 - 策略端到端核心流程总览 (25%)
    # =========================================================================
    s = SlideSpec(4, total)
    s.add_header_footer("PART 02  |  策略核心流程：选股、择时、配置、风控 (25%)", "四模块协同闭环：将原始市场信息稳健转化为低摩擦、合规的实盘持仓")

    s.add_image(56, 102, 1168, 410, FIG_DIR / "00_strategy_architecture_flow.png", border=C_RULE)

    # Bottom 3 coordination takeaway cards
    coord = [
        ("为什么四模块缺一不可？", C_NAVY,
         "单有高 RankIC 选股模型若直接窄幅集中持股（Top-50），会被高换手与高跟踪误差吞噬；必须与 Soft-Tilt 宽基配置、EMA 平滑和合规执行协同，才能把预测力转化为稳定净超额。"),
        ("数据防泄露贯通全流程", C_CRIMSON,
         "成分股周度 PIT、基本面公告日实时反推、训练与测试间强制 10 日隔离期、周五信号下周一开盘撮合——从源头杜绝回测虚高。"),
        ("全流程均以投资问题为导向", C_GREEN,
         "选股解决“买谁/避谁”，择时与滚动解决“何时调/风格变了怎么办”，配置解决“买多少”，风控解决“能不能成交与风险上限”。"),
    ]
    for i, (ct, cc, cb) in enumerate(coord):
        cx = 56 + i * 396
        s.add_rect(cx, 528, 376, 138, fill=C_BG_CARD, border=C_BORDER, rounded=True)
        s.add_rect(cx, 528, 5, 138, fill=cc)
        s.add_text(cx + 16, 538, 346, 24, ct, size=11.2, color=cc, bold=True)
        s.add_text(cx + 16, 566, 346, 92, cb, size=9.8, color=C_MUTED, line_spacing=1.30)
    slides.append(s)

    # =========================================================================
    # SLIDE 5: PART 2 - 选股机制与隐式择时/风格适应 (25%)
    # =========================================================================
    s = SlideSpec(5, total)
    s.add_header_footer("PART 02  |  策略核心流程：选股与择时 (25%)", "选股：TS-CSNet 三步完成横向排序；择时：放弃主观大盘择时，聚焦风格自适应")

    # Left Column: Stock Selection (3 intuitive steps)
    s.add_rect(56, 108, 610, 554, fill=C_WHITE, border=C_BORDER, rounded=True)
    s.add_rect(56, 108, 610, 42, fill=C_NAVY)
    s.add_text(74, 116, 570, 26, "选股引擎：用金融直觉理解 TS-CSNet（14,977 轻量参数）", size=12.5, color=C_WHITE, bold=True)

    steps = [
        ("Step 1  提取 20 日量价走势（GRU 时序模块）", C_STEEL,
         "同样 20 天涨 5%，“缩量缓涨”与“爆量冲高回落”含义截然不同。轻量时序层自动压缩 20 日量价节奏，替代人工硬编几十个易过拟合的技术指标。"),
        ("Step 2  动态平衡量价与基本面（门控融合层）", C_CRIMSON,
         "通过可观测的门控权重 g ∈ (0,1) 动态融合“短期量价”与“静态估值/盈利”。历史均值 g=0.48（约 48% 量价 + 52% 基本面），打破深度学习黑盒印象。"),
        ("Step 3  全市场横向比强弱 + 无偏排序优化", C_GREEN,
         "将当日 500 只成分股放入截面自注意力层互相比较强弱；摒弃易受涨跌停扭曲的 MSE，直接用无偏相关性 + 成对排序损失对齐选股前后顺序。"),
    ]
    sy = 168
    for st, sc, sb in steps:
        s.add_rect(76, sy, 570, 144, fill=C_BG_CARD, border=C_BORDER, rounded=True)
        s.add_rect(76, sy, 6, 144, fill=sc)
        s.add_text(94, sy + 12, 536, 26, st, size=11.5, color=sc, bold=True)
        s.add_text(94, sy + 42, 536, 92, sb, size=10.2, color=C_INK, line_spacing=1.32)
        sy += 160

    # Right Column: Market Timing & Regime Adaptation
    s.add_rect(688, 108, 536, 554, fill=C_WHITE, border=C_BORDER, rounded=True)
    s.add_rect(688, 108, 536, 42, fill=C_CRIMSON)
    s.add_text(706, 116, 500, 26, "择时与适应：为何不做大盘择时？如何应对周期切换？", size=12.5, color=C_WHITE, bold=True)

    timing_blocks = [
        ("为什么不做主动大盘仓位择时？", C_BG_CRIMSON, C_CRIMSON,
         "中证 500 指增基金的核心定位是提供稳定权益 Beta + 持续 Alpha。A 股年度涨幅高度集中在少数暴涨周，择时减仓极易造成不可逆的踏空损失，故维持约 99% 高仓位。"),
        ("双周调仓 + 因果 EMA 信号平滑", C_BG_CARD, C_NAVY,
         "每两周（约 10 交易日）再平衡一次，并对相邻两期打分做 50%/50% 因果平滑：既保留短期量价反转 Alpha，又将年化单边换手压低至 2.69x。"),
        ("年度滚动重训 + 门控风格自适应", C_BG_GREEN, C_GREEN,
         "每年用最近 3 年样本滚动更新模型（末尾 24 周早停验证），随市场演进刷新定价规律：如 2023–2025 重训版已在 2026-11 实盘中自动转向景气光通信与低估电子龙头。"),
    ]
    ty = 168
    for tt, tbg, tc, tb in timing_blocks:
        s.add_rect(708, ty, 496, 144, fill=tbg, border=C_BORDER, rounded=True)
        s.add_text(724, ty + 12, 464, 26, tt, size=11.5, color=tc, bold=True)
        s.add_text(724, ty + 42, 464, 92, tb, size=10.2, color=C_INK, line_spacing=1.32)
        ty += 160
    slides.append(s)

    # =========================================================================
    # SLIDE 6: PART 2 - Soft-Tilt配置案例与风控体系 (25%)
    # =========================================================================
    s = SlideSpec(6, total)
    s.add_header_footer("PART 02  |  策略核心流程：配置与风控 (25%)", "Soft-Tilt 软倾斜配置透明可控，搭配实盘硬约束风控（同时坦诚当前风控短板）")

    s.add_text(56, 102, 1168, 28, "一、Soft-Tilt 权重调整实案（逻辑：锚定流通市值基准权重 × clip(1 + 截面得分 z, 0, 2.5)，摘自 2026-11-24 实盘输出）", size=11.5, color=C_NAVY, bold=True)
    s.add_table(
        x=56, y=134, w=1168,
        col_widths=[102, 160, 92, 116, 84, 152, 92, 370],
        headers=["配置类型", "股票（行业）", "基准权重", "排名 (得分)", "倾斜倍数", "建仓权重 (股数)", "主动偏离", "背后的金融投资逻辑"],
        rows=[
            ["典型超配", "长飞光纤 (通信)", "0.84%", "#7 (+1.45σ)", "2.45×", "1.87% (900股)", "+1.03%", "ROE 34.7%、ΔROE +13.6%，20日回调 -28% 具强修复赔率"],
            ["典型低配", "康龙化成 (医药)", "0.58%", "#422 (-0.82σ)", "0.18×", "0.08% (200股)", "-0.50%", "PE 42.8x 偏高、盈利增速停滞，近期反弹后压缩权重"],
            ["清零排雷", "华虹宏力 (电子)", "0.84%", "#500 (-5.14σ)", "0.00×", "0.00% (0股)", "-0.84%", "PE 309x、ROE 仅 2.8%，20日暴涨 +40% 且高换手 6.7%"],
        ],
        row_h=52.0, header_h=34.0, font_size=9.5,
        highlight_rows={0: C_BG_GREEN, 2: C_BG_CRIMSON},
        col_aligns=["center", "left", "center", "center", "center", "center", "center", "left"],
    )

    # Bottom split: Current Risk Controls vs Honest Gaps
    s.add_rect(56, 354, 668, 308, fill=C_BG_CARD, border=C_BORDER, rounded=True)
    s.add_rect(56, 354, 668, 38, fill=C_GREEN)
    s.add_text(74, 361, 630, 24, "二、已落地的四道风控与实盘合规防线", size=11.8, color=C_WHITE, bold=True)
    rc_items = [
        "• 严格防未来函数：周度 PIT 成分股 + 公告日估值反推 + 训练/测试间强制 10 日隔离期（4/4 单元测试固化）。",
        "• 真实交易硬约束：T+1 开盘撮合；停牌/ST/涨停禁买、跌停/停牌锁仓；零杠杆运作。",
        "• 板块整手与现金风控：主板/创业板 100 股整数倍、科创板 ≥200 股申报门槛 + 贪心补手压低现金至 0.50%。",
        "• 集中度与换手控制：单股 2.5× 上限（历史单股均值最大 1.00%、有效持股 270 只），年化单边换手仅 2.69x。",
    ]
    ry = 408
    for item in rc_items:
        s.add_text(72, ry, 636, 56, item, size=10.3, color=C_INK, line_spacing=1.30)
        ry += 60

    s.add_rect(746, 354, 478, 308, fill=C_BG_CRIMSON, border=C_CRIMSON, border_pt=1.0, rounded=True)
    s.add_rect(746, 354, 478, 38, fill=C_CRIMSON)
    s.add_text(764, 361, 442, 24, "三、当前风控体系的明确不足（诚实审视）", size=11.8, color=C_WHITE, bold=True)
    gap_items = [
        "1. 缺少行业中性硬约束\n历史回测对高波动电子行业平均低配 -2.80%，在 2025 科技单边行情中承受 -4.22% 行业配置拖累。",
        "2. 缺少组合 Beta 下限与风格偏离上限\n模型自然偏好低波防御（全周期 Beta=0.838，2025 年 vol 暴露达 -0.73），导致单边暴涨牛市明显跑输。",
        "→ 结项迭代目标：保持选股核心不变，引入行业内软倾斜标准化与组合 Beta≥0.92 轻约束。",
    ]
    gy = 408
    for idx_g, gtxt in enumerate(gap_items):
        s.add_text(764, gy, 444, 78, gtxt, size=10.1, color=C_NAVY if idx_g == 2 else C_INK, bold=(idx_g == 2), line_spacing=1.28)
        gy += 84
    slides.append(s)

    # =========================================================================
    # SLIDE 7: PART 3 - 回测设计与9年样本外绩效总览 (40% - 1/5)
    # =========================================================================
    s = SlideSpec(7, total)
    s.add_header_footer("PART 03  |  回测与结果分析：回测设计与绩效总览 (40%)", "9年严格样本外：RankIC +7.63%（胜率70.5%），扣真实税费年化净超额 +1.66%")

    s.add_kpi_card(56, 102, 276, 98, "样本外双周 RankIC", "+7.63%", "年化 ICIR +2.37 · 胜率 70.5% (全组第1)", C_NAVY)
    s.add_kpi_card(352, 102, 276, 98, "9年年化净收益 / 超额", "+4.95% / +1.66%", "基准 H00905 +3.25% · 累计净值 1.54", C_GREEN)
    s.add_kpi_card(648, 102, 276, 98, "全周期 IR / 相对回撤", "+0.19 / -14.62%", "前8年累计超额超+33% · 2025单年回撤", C_CRIMSON)
    s.add_kpi_card(944, 102, 280, 98, "年化单边换手率", "2.69x", "全组最低 · 真实分段税费下依然稳健", C_STEEL)

    # Left: NAV & Excess curve image
    s.add_image(56, 214, 770, 456, FIG_DIR / "02_nav_and_excess_curves.png", border=C_RULE)

    # Right: Rigorous Backtest Protocol & Key Takeaways
    s.add_rect(844, 214, 380, 218, fill=C_BG_CARD, border=C_BORDER, rounded=True)
    s.add_text(860, 224, 348, 26, "严格防泄露回测规范", size=11.8, color=C_NAVY, bold=True)
    s.add_text(860, 254, 348, 170,
               "• 样本外区间：2017–2025（9年，230期双周）\n"
               "• 滚动隔离：3年滚动训练 + 24周早停 + 10日隔离期\n"
               "• 严守基准：中证 500 全收益 H00905 + 等权对照\n"
               "• 真实摩擦：买10bp，卖20bp(23-08前)/15bp(后)\n"
               "• 实验公平：所有模型共用同特征源与同撮合引擎",
               size=9.7, color=C_INK, line_spacing=1.34)

    s.add_rect(844, 446, 380, 224, fill=C_BG_NAVY_SOFT, border=C_NAVY, border_pt=0.9, rounded=True)
    s.add_text(860, 456, 348, 26, "核心绩效结论（已验证）", size=11.8, color=C_NAVY, bold=True)
    s.add_text(860, 486, 348, 176,
               "1. 2017–2024 连续 8 年超额曲线稳健攀升，2024 年微盘危机期间仍录得 +4.3% 净超额。\n"
               "2. 2018(+8.2%)、2022(+7.6%)、2023(+3.7%)、2024(+4.3%) 表现尤为亮眼。\n"
               "3. 2025 年科技单边牛市跑输 -10.2%，诚实暴露低 Beta 与行业偏离短板（后页深挖）。",
               size=9.7, color=C_INK, line_spacing=1.32)
    slides.append(s)

    # =========================================================================
    # SLIDE 8: PART 3 - 从0到1的定量迭代优化与消融验证 (40% - 2/5)
    # =========================================================================
    s = SlideSpec(8, total)
    s.add_header_footer("PART 03  |  回测与结果分析：从 0 到 1 迭代优化 (40%)", "四轮定量迭代：发现并攻克“指数排序波动率陷阱”，验证每个模块的真实增益")

    # Left: 4-step iterative evolution timeline
    iters = [
        ("v0 基线建立（Ridge / MLP）", C_STEEL, C_BG_CARD,
         "问题：线性模型难捕捉非线性交互与走势形态。\n结果：M1 RankIC +4.95%，M2 +5.88%，超额偏薄。"),
        ("v1 首次尝试踩坑（指数 ListNet）", C_RED, C_BG_CRIMSON,
         "问题：照搬信息检索 softmax(y/0.5) 触发 Jensen 波动率陷阱，模型狂买高波彩票股（vol暴露飙至+0.62）。\n结果：RankIC 反向跌至 -4.63%，相对回撤 -55%！"),
        ("v2 无偏排序 + TS-CSNet 定型", C_NAVY, C_BG_NAVY_SOFT,
         "改进：改用对称排名目标 + 无偏 PearsonIC & Pairwise RankNet，加入门控与截面注意力。\n结果：RankIC 飙至 +7.63%（ICIR +2.37），9/9年验证集全正。"),
        ("v3 配置升级（Top-50 → Soft-Tilt）", C_GREEN, C_BG_GREEN,
         "改进：由窄幅 Top-50（超额+0.71%/回撤-22.9%）升级为流通市值锚定 Soft-Tilt 宽基排雷。\n结果：年化净超额翻倍至 +1.66%，换手降至 2.69x。"),
    ]
    iy = 106
    for it_t, it_c, it_bg, it_b in iters:
        s.add_rect(56, iy, 540, 128, fill=it_bg, border=C_BORDER, rounded=True)
        s.add_rect(56, iy, 6, 128, fill=it_c)
        s.add_text(74, iy + 10, 508, 24, it_t, size=11.3, color=it_c, bold=True)
        s.add_text(74, iy + 36, 508, 86, it_b, size=9.8, color=C_INK, line_spacing=1.30)
        iy += 140

    # Right: Ablation Overview chart + summary note
    s.add_image(616, 106, 608, 398, FIG_DIR / "01_ablation_overview.png", border=C_RULE)
    s.add_rect(616, 518, 608, 148, fill=C_BG_CARD, border=C_BORDER, rounded=True)
    s.add_text(632, 528, 576, 24, "递进消融实验的核心量化发现", size=11.5, color=C_NAVY, bold=True)
    s.add_text(632, 556, 576, 102,
               "• 单改网络不改损失收益有限：M3 GRU(MSE) 的 RankIC (+5.70%) 甚至略低于 M2 MLP，说明 MSE 限制了时序模型的排序潜力。\n"
               "• 损失函数对齐是最大跃升：M3→M4 仅替换为无偏排序损失，RankIC 即跃升 +1.78% 至 +7.48%；M4→M5 加入门控与截面注意力后，RankIC、净超额与最低换手全面登顶。",
               size=9.7, color=C_INK, line_spacing=1.30)
    slides.append(s)

    # =========================================================================
    # SLIDE 9: PART 3 - 持仓特征与多维业绩归因分析 (40% - 3/5)
    # =========================================================================
    s = SlideSpec(9, total)
    s.add_header_footer("PART 03  |  回测与结果分析：持仓与归因分析 (40%)", "收益归因：行业内选股（+2.36%/年）连胜8年，清零尾部83只股贡献约59%毛超额")

    # Left: Brinson Industry Attribution Figure
    s.add_image(56, 106, 756, 400, FIG_DIR / "05_brinson_industry_attribution.png", border=C_RULE)

    # Bottom-left: Compact Holding Stats strip
    s.add_rect(56, 520, 756, 146, fill=C_BG_CARD, border=C_BORDER, rounded=True)
    s.add_text(72, 530, 724, 24, "历史持仓结构画像（2017–2025 共 230 期双周调仓均值）", size=11.3, color=C_NAVY, bold=True)
    h_cols = [
        ("期均持股 / 排雷", "416 只 / 清零 83 只", "有效持股数 270.5 只"),
        ("集中度控制", "前十大 8.12%", "单股最大均值仅 1.00%"),
        ("主动份额", "Active Share 30.1%", "紧跟基准且有鲜明主动性"),
        ("交易效率", "双周单边换手 10.4%", "年化 2.69x · 摩擦 <0.8%/年"),
    ]
    for i, (hk, hv, hs) in enumerate(h_cols):
        hx = 72 + i * 184
        s.add_text(hx, 562, 176, 20, hk, size=9.2, color=C_LIGHT_TXT, bold=True)
        s.add_text(hx, 585, 176, 26, hv, size=12.0, color=C_NAVY, bold=True)
        s.add_text(hx, 616, 176, 36, hs, size=8.8, color=C_MUTED)

    # Right: 3 Attribution Decomposition Cards
    attr_cards = [
        ("维度一：Brinson 行业与选股归因", C_NAVY, C_BG_NAVY_SOFT,
         "• 行业内个股选择：+2.36%/年（2017–2024 连续 8 年全正），是绝对核心 Alpha 引擎。\n"
         "• 行业配置效应：+0.39%/年，说明策略并非靠赌单一行业赛道；但 2025 年低配电子/通信单年拖累 -4.22%。"),
        ("维度二：头部超配 vs 尾部排雷归因", C_GREEN, C_BG_GREEN,
         "年化毛超额 +2.04% 拆解：\n"
         "• 主动超配强势股：+0.73%/年（占比 35.8%）\n"
         "• 主动低配与清零：+1.31%/年（占比 64.2%），其中仅清零尾部 83 只差股即贡献 +1.21%（59.3%）！"),
        ("维度三：申万一级行业贡献分布", C_CRIMSON, C_BG_CRIMSON,
         "• 正贡献前五：非银金融 (+0.57%)、医药生物 (+0.45%)、房地产 (+0.22%)、基础化工 (+0.17%)、有色金属 (+0.17%)。\n"
         "• 主要负贡献：电子 (-0.53%，平均低配 -2.80%)、通信 (-0.10%)。"),
    ]
    ay = 106
    for at, ac, abg, ab in attr_cards:
        s.add_rect(832, ay, 392, 176, fill=abg, border=C_BORDER, rounded=True)
        s.add_rect(832, ay, 5, 176, fill=ac)
        s.add_text(848, ay + 10, 362, 24, at, size=11.0, color=ac, bold=True)
        s.add_text(848, ay + 38, 362, 130, ab, size=9.6, color=C_INK, line_spacing=1.30)
        ay += 192
    slides.append(s)

    # =========================================================================
    # SLIDE 10: PART 3 - 专题反思：为什么2025年正RankIC却跑输基准？ (40% - 4/5)
    # =========================================================================
    s = SlideSpec(10, total)
    s.add_header_footer("PART 03  |  回测与结果分析：深度反思与可解释性 (40%)", "核心追问：为什么 2025 年 RankIC 仍为 +3.97%，策略却跑输基准 -10.16%？")

    # Left: Cumulative RankIC, Decile Monotonicity, Gate Interpretability (Figure 04)
    s.add_image(56, 106, 700, 560, FIG_DIR / "04_gate_and_ic_interpretability.png", border=C_RULE)

    # Right: Deep Diagnosis of 2025 & Positive RankIC != Positive Excess
    s.add_rect(776, 106, 448, 364, fill=C_BG_CRIMSON, border=C_CRIMSON, border_pt=1.0, rounded=True)
    s.add_rect(776, 106, 448, 40, fill=C_CRIMSON)
    s.add_text(792, 114, 416, 24, "深度拆解：正 RankIC ≠ 正超额的三重定量原因", size=11.6, color=C_WHITE, bold=True)

    diag_txt = (
        "1. 低 Beta 防御暴露在单边大牛市中的数学折价\n"
        "全周期 CAPM Beta=0.838（纯 Alpha=+1.97%/年）。经历 2022–2024 三年熊市滚动训练后，2025 年低波低换手暴露推至极值（vol_60d=-0.73, turn=-0.64, Beta≈0.80）。当基准暴涨 +32.3% 时，仅 Beta 缺口即拖累约 -6.5%！\n\n"
        "2. 行业偏离叠加赛道分化\n"
        "缺乏行业中性约束，低配高波科技龙头导致 2025 年 Brinson 行业配置单项失血 -4.22%。\n\n"
        "3. 等权排名指标与市值加权收益的结构差异\n"
        "RankIC 衡量全池等权排序（2025 年依然 +3.97% 有效），但当指数涨幅高度集中于少数高估高波权重股时，排雷剔除这些标的会造成市值加权组合跑输。"
    )
    s.add_text(792, 156, 416, 304, diag_txt, size=9.7, color=C_INK, line_spacing=1.28)

    # Bottom-right: Gate interpretability insight
    s.add_rect(776, 486, 448, 180, fill=C_BG_CARD, border=C_BORDER, rounded=True)
    s.add_rect(776, 486, 5, 180, fill=C_NAVY)
    s.add_text(794, 496, 414, 24, "左图补充洞见：十分位单调性与门控可解释性", size=11.0, color=C_NAVY, bold=True)
    s.add_text(794, 524, 414, 134,
               "• 十分位高度单调（D2 +5.2% → D10 -13.2%），证实模型并非过拟合噪音，而是稳定识别出尾部大跌股。\n"
               "• 门控曲线均值 g=0.478：轮动快年份（2019/2021/2024）自动偏向短期量价（g≈0.50+），熊市与估值重构期（2018/2025）自动转向基本面（g≈0.43–0.45）。",
               size=9.6, color=C_INK, line_spacing=1.28)
    slides.append(s)

    # =========================================================================
    # SLIDE 11: PART 3 - 适用性分析与策略边界 (40% - 5/5)
    # =========================================================================
    s = SlideSpec(11, total)
    s.add_header_footer("PART 03  |  回测与结果分析：适用性分析 (40%)", "明确适用边界：熊市/震荡市/高波动优势突出，高Beta单边牛承压，成本耐受至30bp")

    s.add_image(56, 102, 574, 344, FIG_DIR / "06_regime_and_style_applicability.png", border=C_RULE)
    s.add_image(650, 102, 574, 344, FIG_DIR / "07_era_and_cost_applicability.png", border=C_RULE)

    # Bottom 3 Applicability Boundary Cards
    bounds = [
        ("优势适用环境（占 ~80% 时间）", C_GREEN, C_BG_GREEN,
         "• 下跌熊市（14.8%天数）：年化超额 +8.08%，IR +1.82，日胜率 57.1%，抗跌极强。\n"
         "• 震荡整理市（64.8%天数）：年化超额 +2.01%，IR +0.39，稳步复利。\n"
         "• 高波动分化市：年化超额 +2.97%（IR +0.45）；2024微盘危机期仍获 +4.3% 超额。"),
        ("策略承压 / 失效边界（约 20% 时间）", C_CRIMSON, C_BG_CRIMSON,
         "• 单边急涨高 Beta 牛市（60日涨 >+8%）：年化超额 -6.85%（IR -0.84），低波动风格天然落后。\n"
         "• 低波动缩量沉闷市：年化超额 -0.78%，个股分化不足难以覆盖摩擦。\n"
         "• 应对思路：需配合行业中性与 Beta 下限约束修补该边界。"),
        ("交易成本与资金容量适用性", C_NAVY, C_BG_NAVY_SOFT,
         "• 成本极其稳健：0bp(+2.45%) → 10bp(+1.89%) → 真实成本(+1.66%) → 20bp(+1.35%) → 30bp 极端摩擦仍有 +0.80% 正超额。\n"
         "• 资金容量：Soft-Tilt 最适 1,000万–5亿元机构基金；100万以下个人资金宜复用同打分切入精选 Top-30/50。"),
    ]
    for i, (bt, bc, bbg, bb) in enumerate(bounds):
        bx = 56 + i * 396
        s.add_rect(bx, 464, 376, 206, fill=bbg, border=C_BORDER, rounded=True)
        s.add_rect(bx, 464, 376, 34, fill=bc)
        s.add_text(bx + 14, 470, 348, 22, bt, size=10.8, color=C_WHITE, bold=True)
        s.add_text(bx + 14, 504, 348, 160, bb, size=9.4, color=C_INK, line_spacing=1.28)
    slides.append(s)

    # =========================================================================
    # SLIDE 12: PART 4 - 2026-11-24 模拟交易建仓计划与合规校验 (20% - 1/3)
    # =========================================================================
    s = SlideSpec(12, total)
    s.add_header_footer("PART 04  |  模拟交易与详细持仓配置 (20%)", "2026-11-24 模拟建仓计划：1,000 万资金建仓 422 只（99.4%仓位），严格满足板块整手限制")

    s.add_kpi_card(56, 102, 276, 98, "初始模拟总资金", "1,000.00 万元", "基准 000905 收盘 6,112.43", C_NAVY)
    s.add_kpi_card(352, 102, 276, 98, "计划建仓股票市值", "994.00 万元", "股票仓位 99.40% · 持股 422 只", C_GREEN)
    s.add_kpi_card(648, 102, 276, 98, "预估费用 / 剩余现金", "0.99万 / 5.00万", "买入成本 10bp · 现金储备 0.50%", C_STEEL)
    s.add_kpi_card(944, 102, 280, 98, "自动清零排雷股", "78 只", "前十大权重 10.55% · 单股 ≤1.87%", C_CRIMSON)

    # Left: Live paper portfolio chart (Figure 08)
    s.add_image(56, 214, 764, 456, FIG_DIR / "08_live_paper_portfolio.png", border=C_RULE)

    # Right: Compliance & Plan vs Execution distinction
    s.add_rect(838, 214, 386, 134, fill=C_BG_CRIMSON, border=C_CRIMSON, border_pt=1.0, rounded=True)
    s.add_text(854, 224, 354, 24, "口径严格区分声明", size=11.2, color=C_CRIMSON, bold=True)
    s.add_text(854, 252, 354, 90,
               "本页展示为基于 2026-11-24 收盘参考价生成的【模拟建仓计划】（权重与股数已冻结归档）；【实际模拟成交结果】将于次日起按实盘开盘成交价与双周调仓纪律逐期跟踪，并在结项汇报中完整复盘。",
               size=9.5, color=C_INK, line_spacing=1.28)

    s.add_rect(838, 362, 386, 166, fill=C_BG_CARD, border=C_BORDER, rounded=True)
    s.add_text(854, 372, 354, 24, "实际市场限制 100% 合规落地", size=11.2, color=C_NAVY, bold=True)
    s.add_text(854, 400, 354, 122,
               "• 主板/创业板：严格 100 股整数倍（1手）申报。\n"
               "• 科创板 (688)：严格满足 ≥200 股最低申报门槛（如源杰科技 300 股、仕佳光子 900 股、惠泰医疗 200 股）。\n"
               "• 贪心余钱补手：将原始向下取整的 7.36% 现金拖累精准压缩至 0.50%，零杠杆。",
               size=9.4, color=C_INK, line_spacing=1.28)

    s.add_rect(838, 542, 386, 128, fill=C_BG_GREEN, border=C_GREEN, border_pt=0.9, rounded=True)
    s.add_text(854, 552, 354, 24, "滚动重训的实盘适应亮点", size=11.2, color=C_GREEN, bold=True)
    s.add_text(854, 580, 354, 84,
               "纳入 2025 样本滚动重训后，模型不再僵化低配科技：实盘对通信超配 +3.95%、对电子近标配 17.03%（行业内持仓 48 只优质股、清零 19 只高估/亏损股），门控均值 g=0.45。",
               size=9.4, color=C_INK, line_spacing=1.28)
    slides.append(s)

    # =========================================================================
    # SLIDE 13: PART 4 - 典型超配、低配与清零剔除标的解析 (20% - 2/3)
    # =========================================================================
    s = SlideSpec(13, total)
    s.add_header_footer("PART 04  |  模拟交易与详细持仓配置 (20%)", "实盘个股配置理由：超配高盈利超跌龙头，低配盈利转弱股，清零高估过热泡沫")

    s.add_table(
        x=56, y=104, w=1168,
        col_widths=[74, 146, 74, 118, 92, 102, 562],
        headers=["类别", "股票 (行业)", "参考价", "权重 (主动)", "计划股数", "排名 (得分)", "结合系统输出与纯 PIT 指标的配置理由"],
        rows=[
            ["超配", "长飞光纤 (通信)", "207.70", "1.87% (+1.03%)", "900 股", "#7 (+1.45)", "ROE 34.7% 且 ΔROE +13.6%，PE 26.6x 估值合理，20日回调 -28% 具强修复弹性；g=0.45"],
            ["超配", "源杰科技 (电子·科创)", "599.30", "1.80% (+0.96%)", "300 股", "#22 (+1.21)", "合规≥200股；ROE 31.7%、ΔROE +17.6% 景气加速，20日深调 -28.1% 给出介入窗口；g=0.45"],
            ["超配", "德明利 (电子)", "168.66", "0.67% (+0.32%)", "400 股", "#41 (+1.03)", "极致盈利估值比（PE 5.7x, ROE 78.7%, ΔROE +16.7%），20日回调 -24.7% 具高安全边际"],
            ["超配", "越秀资本 (非银)", "6.52", "0.59% (+0.32%)", "9,000 股", "#8 (+1.42)", "破净低估（PE 8.3x, PB 0.98x, ROE 11.8%），20日均换手仅 0.35%，提供防御压舱石"],
            ["低配", "康龙化成 (医药)", "41.21", "0.08% (-0.50%)", "200 股", "#422 (-0.82)", "PE 42.8x 偏高、盈利零改善（ΔROE +0.1%），近20日逆势反弹 +10.6% 后大幅减配"],
            ["低配", "赤峰黄金 (有色)", "33.60", "0.20% (-0.35%)", "600 股", "#391 (-0.55)", "近60日盈利明显回落（ΔROE -2.7%）、PB 5.03x 偏高，得分落入后 22%，压减六成权重"],
            ["清零", "华虹宏力 (电子·科创)", "209.39", "0.00% (-0.84%)", "0 股", "#500 (-5.14)", "PE 309x、ROE 仅 2.8%，20日暴涨 +40.0% 且均换手 6.71%，高估过热触发 -1σ 直接清零"],
            ["清零", "燕东微 (电子·科创)", "35.94", "0.00% (-0.39%)", "0 股", "#499 (-4.16)", "持续亏损（PE -46.7x, ROE -6.1%）且盈利继续恶化（ΔROE -3.6%），20日反抽 +17% 果断排雷"],
        ],
        row_h=54.0, header_h=34.0, font_size=9.2,
        highlight_rows={0: C_BG_GREEN, 1: C_BG_GREEN, 2: C_BG_GREEN, 3: C_BG_GREEN, 4: C_BG_CARD, 5: C_BG_CARD, 6: C_BG_CRIMSON, 7: C_BG_CRIMSON},
        col_aligns=["center", "left", "right", "center", "right", "center", "left"],
    )

    s.add_rect(56, 586, 1168, 78, fill=C_BG_NAVY_SOFT, border=C_NAVY, border_pt=0.9, rounded=True)
    s.add_text(74, 595, 1132, 62,
               "核心亮点——同一热门行业内部的“去伪存真”：在第一大行业【电子】中，模型并未盲目追高或一刀切回避，而是买入低估高盈利的德明利/君正股份/源杰科技等 48 只标的（17.03%），同时清零华虹宏力/燕东微/芯源微/中科飞测等 19 只高估亏损股（规避 6.41% 基准风险），将 Brinson 行业内选股优势落地到实盘。完整 422 只清单已归档于 paper_portfolio_live_full.csv。",
               size=9.8, color=C_NAVY, bold=True, line_spacing=1.30)
    slides.append(s)

    # =========================================================================
    # SLIDE 14: PART 4 - 结项模拟复盘机制与下一步迭代路线 (20% - 3/3)
    # =========================================================================
    s = SlideSpec(14, total)
    s.add_header_footer("PART 04  |  模拟跟踪与结项迭代计划 (20%)", "明确研究边界：严守双周模拟跟踪复盘，结项聚焦行业中性与 Beta 风控升级")

    cols = [
        ("一、已验证的中期结论", C_GREEN, C_BG_GREEN,
         "1. 无偏排序选股显著有效\n"
         "TS-CSNet 9年样本外 RankIC +7.63%（ICIR +2.37，胜率 70.5%），显著优于传统线性岭回归与 MSE 深度基线。\n\n"
         "2. 超额来源清晰可验证\n"
         "Brinson 行业内选股（+2.36%/年）连胜 8 年，清零尾部 83 只差股贡献 59% 毛超额；熊市与震荡市优势突出。\n\n"
         "3. 低换手与高实盘合规性\n"
         "年化换手仅 2.69x，30bp 极端成本仍正超额；实盘建仓严格满足主板/科创板整手约束。"),
        ("二、模拟交易跟踪与结项复盘", C_NAVY, C_BG_NAVY_SOFT,
         "跟踪周期：2026-11-25 开盘 → 结项汇报日\n"
         "执行纪律：严格双周再平衡，逐笔记录真实开盘成交价、税费与净值曲线。\n\n"
         "结项汇报三大定量复盘维度：\n"
         "• 重仓检验：复盘前 15 大超配股的实际盈亏贡献；\n"
         "• 排雷检验：核验本期清零的 78 只尾部股后续是否跑输中证 500；\n"
         "• 实时归因：对模拟期实际超额开展 Brinson 行业与选股拆解。"),
        ("三、结项重点优化方向（待验证）", C_CRIMSON, C_BG_CRIMSON,
         "针对 2025 年高 Beta 单边牛市跑输（-10.16%）的核心短板，结项阶段拟在保持选股引擎不变下验证两项轻风控升级：\n\n"
         "1. 行业内 Soft-Tilt 标准化\n"
         "将超配与排雷更多约束在申万一级行业内部，消除电子/通信等单一赛道大幅偏离风险。\n\n"
         "2. 组合 Beta 下限约束（Beta ≥ 0.92）\n"
         "在保留熊市排雷保护的同时，显著缓解单边牛市因 Beta 折价带来的相对回撤。"),
    ]
    for i, (ct, cc, cbg, cb) in enumerate(cols):
        cx = 56 + i * 396
        s.add_rect(cx, 108, 376, 482, fill=cbg, border=C_BORDER, rounded=True)
        s.add_rect(cx, 108, 376, 42, fill=cc)
        s.add_text(cx + 16, 116, 344, 26, ct, size=11.8, color=C_WHITE, bold=True)
        s.add_text(cx + 16, 162, 344, 414, cb, size=10.0, color=C_INK, line_spacing=1.32)

    s.add_rect(56, 608, 1168, 58, fill=C_NAVY, rounded=True)
    s.add_text(56, 616, 1168, 42, "感谢老师与各位同学聆听！欢迎批评指正  |  代码、回测数据与 422 只模拟建仓清单已开源归档至 GitHub: wwc76934-cyber/TS-CSNet", size=11.0, color=C_WHITE, bold=True, align="center", valign="middle")
    slides.append(s)

    return slides


def _xml_bytes(el: etree._Element) -> bytes:
    return etree.tostring(el, xml_declaration=True, encoding="UTF-8", standalone=True)


def write_pptx(slides: List[SlideSpec], out_path: Path) -> None:
    """Compile all SlideSpec objects into a 100% standards-compliant OOXML .pptx."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Save 1:1 PNG previews first
    for sl in slides:
        sl._img.save(PREVIEW_DIR / f"slide_{sl.slide_no:02d}.png")

    # Collect unique media images
    media_map = {}  # Path -> media filename
    media_counter = 1
    for sl in slides:
        for _rid, img_p in sl.images:
            if img_p not in media_map:
                media_map[img_p] = f"image{media_counter}.png"
                media_counter += 1

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. [Content_Types].xml
        ct = etree.Element("Types", xmlns=NS_CT)
        etree.SubElement(ct, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
        etree.SubElement(ct, "Default", Extension="xml", ContentType="application/xml")
        etree.SubElement(ct, "Default", Extension="png", ContentType="image/png")
        etree.SubElement(ct, "Override", PartName="/ppt/presentation.xml", ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml")
        etree.SubElement(ct, "Override", PartName="/ppt/slideMasters/slideMaster1.xml", ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml")
        etree.SubElement(ct, "Override", PartName="/ppt/slideLayouts/slideLayout1.xml", ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml")
        etree.SubElement(ct, "Override", PartName="/ppt/theme/theme1.xml", ContentType="application/vnd.openxmlformats-officedocument.theme+xml")
        etree.SubElement(ct, "Override", PartName="/ppt/presProps.xml", ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml")
        etree.SubElement(ct, "Override", PartName="/ppt/viewProps.xml", ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml")
        etree.SubElement(ct, "Override", PartName="/ppt/tableStyles.xml", ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml")
        for sl in slides:
            etree.SubElement(ct, "Override", PartName=f"/ppt/slides/slide{sl.slide_no}.xml", ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        zf.writestr("[Content_Types].xml", _xml_bytes(ct))

        # 2. _rels/.rels
        rels = etree.Element("Relationships", xmlns=NS_REL)
        etree.SubElement(rels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", Target="ppt/presentation.xml")
        zf.writestr("_rels/.rels", _xml_bytes(rels))

        # 3. ppt/presentation.xml & rels
        pres = etree.Element(f"{{{NS_P}}}presentation", nsmap={"a": NS_A, "r": NS_R, "p": NS_P})
        smIdLst = etree.SubElement(pres, f"{{{NS_P}}}sldMasterIdLst")
        etree.SubElement(smIdLst, f"{{{NS_P}}}sldMasterId", id="2147483648", **{f"{{{NS_R}}}id": "rId1"})
        sldIdLst = etree.SubElement(pres, f"{{{NS_P}}}sldIdLst")
        for idx, sl in enumerate(slides):
            etree.SubElement(sldIdLst, f"{{{NS_P}}}sldId", id=str(256 + idx), **{f"{{{NS_R}}}id": f"rId{idx + 5}"})
        etree.SubElement(pres, f"{{{NS_P}}}sldSz", cx=str(W_EMU), cy=str(H_EMU), type="screen16x9")
        etree.SubElement(pres, f"{{{NS_P}}}notesSz", cx="6858000", cy="9144000")
        zf.writestr("ppt/presentation.xml", _xml_bytes(pres))

        prels = etree.Element("Relationships", xmlns=NS_REL)
        etree.SubElement(prels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", Target="slideMasters/slideMaster1.xml")
        etree.SubElement(prels, "Relationship", Id="rId2", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps", Target="presProps.xml")
        etree.SubElement(prels, "Relationship", Id="rId3", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps", Target="viewProps.xml")
        etree.SubElement(prels, "Relationship", Id="rId4", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles", Target="tableStyles.xml")
        for idx, sl in enumerate(slides):
            etree.SubElement(prels, "Relationship", Id=f"rId{idx + 5}", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", Target=f"slides/slide{sl.slide_no}.xml")
        zf.writestr("ppt/_rels/presentation.xml.rels", _xml_bytes(prels))

        # 4. Boilerplate presProps, viewProps, tableStyles
        zf.writestr("ppt/presProps.xml", b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>')
        zf.writestr("ppt/viewProps.xml", b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr></p:viewPr>')
        zf.writestr("ppt/tableStyles.xml", b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>')

        # 5. Theme
        theme_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="{NS_A}" name="PKU Academic Minimalist">
  <a:themeElements>
    <a:clrScheme name="PKU">
      <a:dk1><a:srgbClr val="{C_INK}"/></a:dk1>
      <a:lt1><a:srgbClr val="{C_WHITE}"/></a:lt1>
      <a:dk2><a:srgbClr val="{C_NAVY}"/></a:dk2>
      <a:lt2><a:srgbClr val="{C_BG_CARD}"/></a:lt2>
      <a:accent1><a:srgbClr val="{C_NAVY}"/></a:accent1>
      <a:accent2><a:srgbClr val="{C_CRIMSON}"/></a:accent2>
      <a:accent3><a:srgbClr val="{C_GREEN}"/></a:accent3>
      <a:accent4><a:srgbClr val="{C_STEEL}"/></a:accent4>
      <a:accent5><a:srgbClr val="{C_RED}"/></a:accent5>
      <a:accent6><a:srgbClr val="{C_MUTED}"/></a:accent6>
      <a:hlink><a:srgbClr val="{C_STEEL}"/></a:hlink>
      <a:folHlink><a:srgbClr val="{C_CRIMSON}"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="PKU YaHei">
      <a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Microsoft YaHei"/></a:majorFont>
      <a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Microsoft YaHei"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Minimalist">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>'''.encode("utf-8")
        zf.writestr("ppt/theme/theme1.xml", theme_xml)

        # 6. SlideMaster & SlideLayout (clean blank master/layout)
        master_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>'''.encode("utf-8")
        zf.writestr("ppt/slideMasters/slideMaster1.xml", master_xml)

        mrels = etree.Element("Relationships", xmlns=NS_REL)
        etree.SubElement(mrels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", Target="../slideLayouts/slideLayout1.xml")
        etree.SubElement(mrels, "Relationship", Id="rId2", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", Target="../theme/theme1.xml")
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", _xml_bytes(mrels))

        layout_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}" type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>'''.encode("utf-8")
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml)

        lrels = etree.Element("Relationships", xmlns=NS_REL)
        etree.SubElement(lrels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", Target="../slideMasters/slideMaster1.xml")
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", _xml_bytes(lrels))

        # 7. Media files
        for img_p, mname in media_map.items():
            zf.write(img_p, f"ppt/media/{mname}")

        # 8. Individual Slides
        for sl in slides:
            sld = etree.Element(f"{{{NS_P}}}sld", nsmap={"a": NS_A, "r": NS_R, "p": NS_P})
            cSld = etree.SubElement(sld, f"{{{NS_P}}}cSld")
            bg = etree.SubElement(cSld, f"{{{NS_P}}}bg")
            bgPr = etree.SubElement(bg, f"{{{NS_P}}}bgPr")
            sf = etree.SubElement(bgPr, f"{{{NS_A}}}solidFill")
            etree.SubElement(sf, f"{{{NS_A}}}srgbClr", val=sl.bg_color)
            etree.SubElement(bgPr, f"{{{NS_A}}}effectLst")

            spTree = etree.SubElement(cSld, f"{{{NS_P}}}spTree")
            nvGrp = etree.SubElement(spTree, f"{{{NS_P}}}nvGrpSpPr")
            etree.SubElement(nvGrp, f"{{{NS_P}}}cNvPr", id="1", name="")
            etree.SubElement(nvGrp, f"{{{NS_P}}}cNvGrpSpPr")
            etree.SubElement(nvGrp, f"{{{NS_P}}}nvPr")
            grpSpPr = etree.SubElement(spTree, f"{{{NS_P}}}grpSpPr")
            xfrm = etree.SubElement(grpSpPr, f"{{{NS_A}}}xfrm")
            etree.SubElement(xfrm, f"{{{NS_A}}}off", x="0", y="0")
            etree.SubElement(xfrm, f"{{{NS_A}}}ext", cx="0", cy="0")
            etree.SubElement(xfrm, f"{{{NS_A}}}chOff", x="0", y="0")
            etree.SubElement(xfrm, f"{{{NS_A}}}chExt", cx="0", cy="0")

            for sh in sl.shapes_xml:
                spTree.append(sh)

            clrOvr = etree.SubElement(sld, f"{{{NS_P}}}clrMapOvr")
            etree.SubElement(clrOvr, f"{{{NS_A}}}masterClrMapping")
            zf.writestr(f"ppt/slides/slide{sl.slide_no}.xml", _xml_bytes(sld))

            srels = etree.Element("Relationships", xmlns=NS_REL)
            etree.SubElement(srels, "Relationship", Id="rIdLayout", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", Target="../slideLayouts/slideLayout1.xml")
            for rid, img_p in sl.images:
                etree.SubElement(srels, "Relationship", Id=rid, Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", Target=f"../media/{media_map[img_p]}")
            zf.writestr(f"ppt/slides/_rels/slide{sl.slide_no}.xml.rels", _xml_bytes(srels))

    print(f"Successfully generated {len(slides)}-slide PPTX: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved 1:1 visual previews to: {PREVIEW_DIR}")


def main() -> None:
    slides = build_all_slides()
    write_pptx(slides, OUT_PPTX)


if __name__ == "__main__":
    main()
