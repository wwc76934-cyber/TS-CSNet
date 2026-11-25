"""Build the redesigned, spacious 26-slide PKU Executive Academic Midterm Deck (.pptx)
for「中证500量价与基本面融合增强策略」.

Design highlights:
  - Spacious visual pacing: every major backtest/attribution/applicability chart gets
    its own dedicated large hero panel (60-90% of slide width) with 11-13pt readable labels.
  - Soft-alabaster canvas (#F7F9FC) with crisp floating white cards (#FFFFFF + subtle shadow),
    PKU Crimson (#8B1A24) + Institutional Navy (#1B365D) brand palette, top chapter
    navigation pills, and 4 architectural section-divider slides.
  - 100% natively editable PowerPoint / WPS OOXML shapes, tables, text boxes & charts,
    paired with 1:1 1280x720 PNG previews in figures/ppt_previews/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
from typing import List, Optional, Tuple
import zipfile

from lxml import etree
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(r"D:\QuantProjects\tscs-alpha")
FIG_DIR = PROJECT_ROOT / "figures"
SLIDE_FIG_DIR = FIG_DIR / "slides"
OUT_PPTX = PROJECT_ROOT / "docs" / "中证500量价与基本面融合增强策略_中期汇报PPT.pptx"
PREVIEW_DIR = FIG_DIR / "ppt_previews"

W_PX, H_PX = 1280, 720
EMU = 9525
W_EMU, H_EMU = W_PX * EMU, H_PX * EMU

# Elevated PKU Academic Palette
C_CRIMSON = "8B1A24"
C_CRIMSON_DARK = "69121A"
C_NAVY = "1B365D"
C_NAVY_DARK = "102340"
C_STEEL = "3B6998"
C_GREEN = "1B6B4F"
C_RED = "B91C1C"
C_GOLD = "B47B28"
C_INK = "1E293B"
C_MUTED = "475569"
C_LIGHT_TXT = "64748B"
C_CANVAS = "F7F9FC"
C_WHITE = "FFFFFF"
C_BG_SOFT = "EFF4FA"
C_BG_CRIMSON = "FDF4F5"
C_BG_GREEN = "F1F9F5"
C_BG_GOLD = "FEF9EE"
C_BORDER = "D8E1EB"
C_SHADOW = "E6ECF2"
C_RULE = "DCE4EE"

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CT = "http://schemas.openxmlformats.org/package/content-types"

CHAPTERS = [
    (1, "01 投资思想"),
    (2, "02 核心流程"),
    (3, "03 回测归因"),
    (4, "04 模拟交易"),
]


def _hex_rgb(h: str) -> Tuple[int, int, int]:
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _pil_font(pt: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    px = max(10, int(round(pt * 1.333)))
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, px)


_TOKEN_RE = re.compile(r"[A-Za-z0-9Δ+.\-%/≥≤≈×~]+|.")
_CLOSE_PUNCT = set("，。；：！？、）》】%]")


def _wrap_text_pil(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
    lines: List[str] = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue
        tokens = _TOKEN_RE.findall(raw_line)
        merged: List[str] = []
        for tk in tokens:
            if merged and tk in _CLOSE_PUNCT:
                merged[-1] += tk
            else:
                merged.append(tk)
        cur = ""
        for tk in merged:
            test = cur + tk
            if draw.textlength(test, font=font) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur.rstrip())
                cur = tk.lstrip() if cur else tk
                while draw.textlength(cur, font=font) > max_w and len(cur) > 1:
                    cut = len(cur) - 1
                    while cut > 1 and draw.textlength(cur[:cut], font=font) > max_w:
                        cut -= 1
                    lines.append(cur[:cut])
                    cur = cur[cut:]
        if cur:
            lines.append(cur)
    return lines


@dataclass
class SlideSpec:
    slide_no: int
    total_slides: int
    bg_color: str = C_CANVAS
    shapes_xml: List[etree._Element] = field(default_factory=list)
    images: List[Tuple[str, Path]] = field(default_factory=list)
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
        self, x: float, y: float, w: float, h: float,
        fill: Optional[str] = None,
        border: Optional[str] = None,
        border_pt: float = 0.75,
        rounded: bool = False,
        radius_px: int = 10,
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
            adj_val = int(min(22000, max(3000, (radius_px / max(1.0, min(w, h))) * 100000)))
            etree.SubElement(avLst, f"{{{NS_A}}}gd", name="adj", fmla=f"val {adj_val}")

        if fill:
            sf = etree.SubElement(spPr, f"{{{NS_A}}}solidFill")
            etree.SubElement(sf, f"{{{NS_A}}}srgbClr", val=fill)
        else:
            etree.SubElement(spPr, f"{{{NS_A}}}noFill")

        if border:
            ln = etree.SubElement(spPr, f"{{{NS_A}}}ln", w=str(int(border_pt * 12700)))
            sfb = etree.SubElement(ln, f"{{{NS_A}}}solidFill")
            etree.SubElement(sfb, f"{{{NS_A}}}srgbClr", val=border)
        else:
            ln = etree.SubElement(spPr, f"{{{NS_A}}}ln")
            etree.SubElement(ln, f"{{{NS_A}}}noFill")

        self.shapes_xml.append(sp)

        bbox = [int(x), int(y), int(x + w), int(y + h)]
        f_rgb = _hex_rgb(fill) if fill else None
        b_rgb = _hex_rgb(border) if border else None
        bw = max(1, int(round(border_pt * 1.2))) if border else 0
        if rounded:
            self._draw.rounded_rectangle(bbox, radius=radius_px, fill=f_rgb, outline=b_rgb, width=bw)
        else:
            self._draw.rectangle(bbox, fill=f_rgb, outline=b_rgb, width=bw)

    def add_card(
        self, x: float, y: float, w: float, h: float,
        fill: str = C_WHITE, border: str = C_BORDER,
        top_accent: Optional[str] = None, left_accent: Optional[str] = None,
        shadow: bool = True, radius_px: int = 10,
    ) -> None:
        if shadow:
            self.add_rect(x + 2, y + 3, w, h, fill=C_SHADOW, rounded=True, radius_px=radius_px)
        self.add_rect(x, y, w, h, fill=fill, border=border, border_pt=0.8, rounded=True, radius_px=radius_px)
        if top_accent:
            self.add_rect(x, y, w, 5, fill=top_accent)
        if left_accent:
            self.add_rect(x, y, 5, h, fill=left_accent)

    def add_text(
        self, x: float, y: float, w: float, h: float,
        text: str, size: float = 12.0, color: str = C_INK,
        bold: bool = False, align: str = "left", valign: str = "top",
        line_spacing: float = 1.28, pad_x: float = 4.0, pad_y: float = 2.0,
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
            wrap="square", anchor=anchor_map.get(valign, "t"),
            lIns=str(int(pad_x * EMU)), rIns=str(int(pad_x * EMU)),
            tIns=str(int(pad_y * EMU)), bIns=str(int(pad_y * EMU)),
        )
        etree.SubElement(txBody, f"{{{NS_A}}}lstStyle")

        algn_map = {"left": "l", "center": "ctr", "right": "r"}
        for para_txt in text.split("\n"):
            p = etree.SubElement(txBody, f"{{{NS_A}}}p")
            pPr = etree.SubElement(p, f"{{{NS_A}}}pPr", algn=algn_map.get(align, "l"))
            lnSpc = etree.SubElement(pPr, f"{{{NS_A}}}lnSpc")
            etree.SubElement(lnSpc, f"{{{NS_A}}}spcPct", val=str(int(line_spacing * 100000)))
            r = etree.SubElement(p, f"{{{NS_A}}}r")
            rPr = etree.SubElement(r, f"{{{NS_A}}}rPr", lang="zh-CN", sz=str(int(size * 100)), b="1" if bold else "0", dirty="0")
            sf = etree.SubElement(rPr, f"{{{NS_A}}}solidFill")
            etree.SubElement(sf, f"{{{NS_A}}}srgbClr", val=color)
            etree.SubElement(rPr, f"{{{NS_A}}}latin", typeface="Microsoft YaHei")
            etree.SubElement(rPr, f"{{{NS_A}}}ea", typeface="Microsoft YaHei")
            t_el = etree.SubElement(r, f"{{{NS_A}}}t")
            t_el.text = para_txt

        self.shapes_xml.append(sp)

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

    def add_chart_card(self, x: float, y: float, w: float, h: float, title: str, img_path: Path, subtitle: str = "", accent: str = C_NAVY) -> None:
        """Draw an elevated white chart container with clean header + large fitted figure."""
        self.add_card(x, y, w, h, fill=C_WHITE, border=C_BORDER, shadow=True, radius_px=10)
        self.add_rect(x + 16, y + 14, 4, 20, fill=accent)
        self.add_text(x + 26, y + 11, w - 40, 26, title, size=11.5, color=C_NAVY, bold=True)
        if subtitle:
            self.add_text(x + 26, y + 34, w - 40, 18, subtitle, size=8.8, color=C_LIGHT_TXT)
            img_y = y + 54
        else:
            img_y = y + 40
        pad = 12
        self.add_image(x + pad, img_y, w - 2 * pad, h - (img_y - y) - pad, img_path)

    def add_image(self, x: float, y: float, max_w: float, max_h: float, img_path: Path) -> None:
        with Image.open(img_path) as im:
            iw, ih = im.size
        scale = min(max_w / iw, max_h / ih)
        fw, fh = iw * scale, ih * scale
        fx = x + (max_w - fw) / 2.0
        fy = y + (max_h - fh) / 2.0

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

        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB").resize((max(1, int(round(fw))), max(1, int(round(fh)))), Image.Resampling.LANCZOS)
            self._img.paste(im_rgb, (int(round(fx)), int(round(fy))))

    def add_table(
        self, x: float, y: float, w: float,
        col_widths: List[float], headers: List[str], rows: List[List[str]],
        row_h: float = 38.0, header_h: float = 38.0, font_size: float = 10.2,
        header_bg: str = C_NAVY, highlight_rows: Optional[dict] = None,
        col_aligns: Optional[List[str]] = None,
    ) -> float:
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
            bg = header_bg if is_hdr else highlight_rows.get(r_idx, C_WHITE if r_idx % 2 == 0 else "F8FAFD")
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
                    if val.startswith("+") and ("%" in val or "σ" in val):
                        cell_fg = C_GREEN
                        cell_bold = True
                    elif val.startswith("-") and ("%" in val or "σ" in val):
                        cell_fg = C_RED
                rPr = etree.SubElement(r, f"{{{NS_A}}}rPr", lang="zh-CN", sz=str(int(font_size * 100)), b="1" if cell_bold else "0")
                sf = etree.SubElement(rPr, f"{{{NS_A}}}solidFill")
                etree.SubElement(sf, f"{{{NS_A}}}srgbClr", val=cell_fg)
                etree.SubElement(rPr, f"{{{NS_A}}}latin", typeface="Microsoft YaHei")
                etree.SubElement(rPr, f"{{{NS_A}}}ea", typeface="Microsoft YaHei")
                t_el = etree.SubElement(r, f"{{{NS_A}}}t")
                t_el.text = val

                tcPr = etree.SubElement(tc, f"{{{NS_A}}}tcPr", anchor="ctr", marL=str(int(7 * EMU)), marR=str(int(7 * EMU)), marT=str(int(4 * EMU)), marB=str(int(4 * EMU)))
                ln = etree.SubElement(tcPr, f"{{{NS_A}}}lnB", w="9525")
                sfb = etree.SubElement(ln, f"{{{NS_A}}}solidFill")
                etree.SubElement(sfb, f"{{{NS_A}}}srgbClr", val=C_BORDER)
                sfill = etree.SubElement(tcPr, f"{{{NS_A}}}solidFill")
                etree.SubElement(sfill, f"{{{NS_A}}}srgbClr", val=bg)

        _add_row_xml(headers, header_h, True, -1)
        for ri, row in enumerate(rows):
            _add_row_xml(row, row_h, False, ri)

        self.shapes_xml.append(gf)

        cur_y = y
        all_rows = [(headers, header_h, True, -1)] + [(r, row_h, False, idx) for idx, r in enumerate(rows)]
        for cells, rh, is_hdr, ri in all_rows:
            bg = header_bg if is_hdr else highlight_rows.get(ri, C_WHITE if ri % 2 == 0 else "F8FAFD")
            cur_x = x
            for ci, val in enumerate(cells):
                cw = col_widths[ci]
                self._draw.rectangle([int(cur_x), int(cur_y), int(cur_x + cw), int(cur_y + rh)], fill=_hex_rgb(bg))
                self._draw.line([(int(cur_x), int(cur_y + rh)), (int(cur_x + cw), int(cur_y + rh))], fill=_hex_rgb(C_BORDER), width=1)
                cell_bold = is_hdr or (ci == 0)
                cell_fg = C_WHITE if is_hdr else C_INK
                if not is_hdr:
                    if val.startswith("+") and ("%" in val or "σ" in val):
                        cell_fg = C_GREEN
                        cell_bold = True
                    elif val.startswith("-") and ("%" in val or "σ" in val):
                        cell_fg = C_RED
                font = _pil_font(font_size, bold=cell_bold)
                lines = _wrap_text_pil(self._draw, val, font, max(20, int(cw - 14)))
                lh = int(round(font_size * 1.333 * 1.18))
                ty = cur_y + max(2, (rh - len(lines) * lh) / 2.0)
                for ln in lines:
                    lw = self._draw.textlength(ln, font=font)
                    if col_aligns[ci] == "center":
                        tx = cur_x + (cw - lw) / 2.0
                    elif col_aligns[ci] == "right":
                        tx = cur_x + cw - 7 - lw
                    else:
                        tx = cur_x + 7
                    self._draw.text((int(tx), int(ty)), ln, font=font, fill=_hex_rgb(cell_fg))
                    ty += lh
                cur_x += cw
            cur_y += rh
        return tot_h

    def add_header_footer(self, active_ch: int, title: str, tag: str = "") -> None:
        # Clean white header band at top with bottom border
        self.add_rect(0, 0, W_PX, 90, fill=C_WHITE)
        self.add_rect(0, 0, W_PX, 4, fill=C_CRIMSON)
        self.add_rect(0, 90, W_PX, 1, fill=C_RULE)

        # Top-right chapter navigation pills
        nx = 772
        for ch_id, ch_name in CHAPTERS:
            is_act = (ch_id == active_ch)
            self.add_rect(
                nx, 14, 110, 22,
                fill=C_CRIMSON if is_act else "F1F5F9",
                rounded=True, radius_px=6,
            )
            self.add_text(
                nx, 14, 110, 22, ch_name,
                size=8.3, color=C_WHITE if is_act else C_LIGHT_TXT,
                bold=is_act, align="center", valign="middle",
            )
            nx += 116

        if tag:
            self.add_text(52, 15, 700, 20, tag, size=9.8, color=C_CRIMSON, bold=True)
            self.add_text(52, 39, 1176, 40, title, size=19.5, color=C_NAVY, bold=True)
        else:
            self.add_text(52, 26, 710, 44, title, size=20.0, color=C_NAVY, bold=True)

        # Footer
        self.add_rect(52, 686, 1176, 1.0, fill=C_RULE)
        self.add_text(52, 691, 820, 20, "北京大学量化投资课程项目中期汇报  ·  中证500量价与基本面融合增强策略", size=8.6, color=C_LIGHT_TXT)
        self.add_text(1090, 691, 138, 20, f"{self.slide_no:02d} / {self.total_slides:02d}", size=9.0, color=C_NAVY, bold=True, align="right")

    def add_kpi_card(
        self, x: float, y: float, w: float, h: float,
        label: str, value: str, sub: str, accent: str = C_NAVY,
    ) -> None:
        self.add_card(x, y, w, h, fill=C_WHITE, border=C_BORDER, top_accent=accent, shadow=True, radius_px=10)
        self.add_text(x + 16, y + 13, w - 28, 20, label, size=9.8, color=C_MUTED, bold=True)
        self.add_text(x + 16, y + 36, w - 28, 36, value, size=20.5, color=accent, bold=True)
        self.add_text(x + 16, y + 74, w - 28, h - 78, sub, size=9.0, color=C_LIGHT_TXT)


def _make_section_slide(
    slide_no: int, total: int, sec_num: str, weight_txt: str,
    title: str, subtitle: str, highlights: List[Tuple[str, str, str]],
) -> SlideSpec:
    """Create an executive architectural section-divider hero slide."""
    s = SlideSpec(slide_no, total, bg_color=C_WHITE)
    # Left architectural navy+crimson panel
    s.add_rect(0, 0, 470, H_PX, fill=C_NAVY)
    s.add_rect(0, 0, 18, H_PX, fill=C_CRIMSON)
    s.add_rect(470, 0, 6, H_PX, fill=C_CRIMSON)

    s.add_text(64, 150, 360, 30, f"SECTION  {sec_num}   |   {weight_txt}", size=11.5, color="F8B4B8", bold=True)
    s.add_text(60, 195, 360, 120, sec_num, size=72.0, color=C_WHITE, bold=True)
    s.add_rect(64, 336, 96, 4, fill=C_CRIMSON)
    s.add_text(64, 362, 368, 90, title, size=24.0, color=C_WHITE, bold=True, line_spacing=1.25)
    s.add_text(64, 472, 368, 96, subtitle, size=11.0, color="CBD5E1", line_spacing=1.38)
    s.add_text(64, 668, 368, 24, "中证500量价与基本面融合增强策略", size=9.2, color="94A3B8")

    # Right clean overview cards
    s.add_rect(476, 0, W_PX - 476, H_PX, fill=C_CANVAS)
    s.add_text(536, 118, 680, 28, "本节核心议题与关键看点", size=13.0, color=C_CRIMSON, bold=True)
    cy = 166
    for idx, (ht, hv, hb) in enumerate(highlights, start=1):
        s.add_card(536, cy, 676, 134, fill=C_WHITE, border=C_BORDER, left_accent=C_NAVY if idx % 2 == 1 else C_CRIMSON, shadow=True)
        s.add_text(560, cy + 16, 368, 26, ht, size=13.0, color=C_NAVY, bold=True)
        s.add_text(934, cy + 14, 258, 30, hv, size=14.0, color=C_CRIMSON if idx % 2 == 0 else C_GREEN, bold=True, align="right")
        s.add_text(560, cy + 52, 632, 68, hb, size=10.5, color=C_MUTED, line_spacing=1.35)
        cy += 156

    s.add_text(1090, 688, 138, 20, f"{slide_no:02d} / {total:02d}", size=9.2, color=C_NAVY, bold=True, align="right")
    return s


def build_all_slides() -> List[SlideSpec]:
    total = 26
    slides: List[SlideSpec] = []

    # =========================================================================
    # 01. COVER
    # =========================================================================
    s = SlideSpec(1, total, bg_color=C_WHITE)
    # Rich architectural left ribbon + subtle top right accent
    s.add_rect(0, 0, 34, H_PX, fill=C_CRIMSON)
    s.add_rect(34, 0, 12, H_PX, fill=C_NAVY)
    s.add_rect(46, 0, W_PX - 46, 6, fill=C_NAVY)
    s.add_rect(46, H_PX - 76, W_PX - 46, 76, fill=C_CANVAS)
    s.add_rect(46, H_PX - 76, W_PX - 46, 1, fill=C_RULE)

    s.add_rect(104, 96, 268, 34, fill=C_BG_CRIMSON, border=C_CRIMSON, border_pt=1.0, rounded=True, radius_px=8)
    s.add_text(104, 98, 268, 30, "北京大学 · 量化投资课程项目中期汇报", size=10.5, color=C_CRIMSON, bold=True, align="center", valign="middle")

    s.add_text(104, 158, 1080, 88, "中证500量价与基本面融合增强策略", size=36.0, color=C_NAVY, bold=True)
    s.add_text(104, 246, 1060, 44, "从投资思想、组合管理、多维实证归因到 1,000 万元实盘模拟建仓", size=17.0, color=C_MUTED)
    s.add_rect(104, 308, 160, 4.5, fill=C_CRIMSON)
    s.add_rect(272, 310, 420, 1.5, fill=C_RULE)

    cards = [
        ("产品定位", "机构指增模拟基金", "1,000 万元初始规模 · 稳健增值", C_NAVY),
        ("投资标的", "中证 500 成分股", "周度 PIT 成分快照 · 零幸存者偏差", C_CRIMSON),
        ("收益双轮", "精选超配 + 尾部排雷", "量价走势融合公告日纯 PIT 基本面", C_GREEN),
        ("当前进展", "9年回测 + 模拟建仓", "2026-11-24 冻结 422 只合规持仓", C_STEEL),
    ]
    for i, (ck, cv, cs, ca) in enumerate(cards):
        cx = 104 + i * 278
        s.add_card(cx, 356, 260, 140, fill=C_CANVAS, border=C_BORDER, top_accent=ca, shadow=True)
        s.add_text(cx + 18, 376, 224, 22, ck, size=10.0, color=C_LIGHT_TXT, bold=True)
        s.add_text(cx + 18, 404, 224, 34, cv, size=15.5, color=C_NAVY, bold=True)
        s.add_text(cx + 18, 446, 224, 36, cs, size=9.6, color=C_MUTED, line_spacing=1.3)

    s.add_rect(104, 532, 1094, 58, fill=C_BG_SOFT, border=C_BORDER, rounded=True, radius_px=8)
    s.add_text(124, 543, 1054, 36, "汇报结构：01 策略背景与投资思想 (15%)   →   02 策略核心流程 (25%)   →   03 回测、归因与适用性分析 (40%)   →   04 模拟交易与结项计划 (20%)", size=10.8, color=C_NAVY, bold=True, valign="middle")

    s.add_text(104, H_PX - 50, 680, 26, "汇报人 / 小组成员：________________        业绩比较基准：中证 500 全收益指数 (H00905)", size=10.5, color=C_MUTED)
    s.add_text(940, H_PX - 50, 258, 26, "2026 年 11 月", size=11.0, color=C_NAVY, bold=True, align="right")
    slides.append(s)

    # =========================================================================
    # 02. AGENDA / ROADMAP
    # =========================================================================
    s = SlideSpec(2, total)
    s.add_header_footer(1, "汇报框架：以金融投资逻辑为主线，以 9 年实证数据与实盘建仓为支撑", "OVERVIEW  |  目录与内容比重")

    ag = [
        ("01", "策略背景与基本投资思想", "比重 15%  ·  第 03–05 页", C_NAVY,
         "• 为什么选择中证 500 指数增强？基金定位与基准设定\n"
         "• 量价、估值与盈利质量三类纯 PIT 因子的金融学逻辑\n"
         "• “头部超配 + 尾部排雷”双引擎与机器学习的辅助价值"),
        ("02", "策略核心流程：选股·择时·配置·风控", "比重 25%  ·  第 06–10 页", C_STEEL,
         "• 选股：深入浅出解析多视角横向排序机制\n"
         "• 择时：坚持 99% 高仓位不赌大盘，通过滚动与门控适应风格\n"
         "• 配置与风控：Soft-Tilt 三股实案、合规防线与现有不足"),
        ("03", "回测、多维归因与适用性分析（核心重点）", "比重 40%  ·  第 11–22 页", C_CRIMSON,
         "• 9年严格样本外净值、超额曲线、分年度表现与 4 轮定量迭代\n"
         "• 持仓画像、十分位排雷归因、Brinson 行业与选股拆解\n"
         "• 深度反思：为何 2025 年正 RankIC 仍跑输？适用环境与成本边界"),
        ("04", "模拟交易配置与结项复盘方案", "比重 20%  ·  第 23–26 页", C_GREEN,
         "• 2026-11-24 实时输出的 1,000 万元模拟建仓计划与合规校验\n"
         "• 超配、低配与清零三类真实标的解析（电子行业内部去伪存真）\n"
         "• 双周实盘跟踪机制、结项复盘维度与风控优化方向"),
    ]
    coords = [(52, 118), (652, 118), (52, 404), (652, 404)]
    for (num, t, wt, col, body), (ax_x, ax_y) in zip(ag, coords):
        s.add_card(ax_x, ax_y, 576, 260, fill=C_WHITE, border=C_BORDER, top_accent=col, shadow=True)
        s.add_rect(ax_x + 22, ax_y + 24, 54, 54, fill=col, rounded=True, radius_px=10)
        s.add_text(ax_x + 22, ax_y + 24, 54, 54, num, size=20.0, color=C_WHITE, bold=True, align="center", valign="middle")
        s.add_text(ax_x + 90, ax_y + 24, 464, 28, t, size=14.5, color=C_NAVY, bold=True)
        s.add_text(ax_x + 90, ax_y + 54, 464, 22, wt, size=10.0, color=col, bold=True)
        s.add_rect(ax_x + 22, ax_y + 92, 532, 1, fill=C_RULE)
        s.add_text(ax_x + 22, ax_y + 106, 532, 140, body, size=10.8, color=C_INK, line_spacing=1.42)
    slides.append(s)

    # =========================================================================
    # 03. SECTION 01 DIVIDER
    # =========================================================================
    slides.append(_make_section_slide(
        3, total, "01", "内容比重 15%",
        "策略背景与基本投资思想",
        "立足中证 500 中盘成长沃土，融合短期量价行为与公告日基本面安全垫，构建机构级纯多头指数增强产品。",
        [
            ("明确基金定位", "1,000 万规模", "机构/公募指增产品定位，对标含分红的中证 500 全收益指数 (H00905)"),
            ("坚守金融本源", "3 大因子维度", "20日量价行为 + 估值安全边际 + 盈利质量改善，100% 公告日纯 PIT 数据"),
            ("明晰模型角色", "辅助而非主导", "引入轻量机器学习刻画非线性交互与横向相对强弱，杜绝黑盒炫技"),
        ],
    ))

    # =========================================================================
    # 04. 基金定位、为什么选中证500与双轮超额逻辑
    # =========================================================================
    s = SlideSpec(4, total)
    s.add_header_footer(1, "聚焦中证 500 中盘成长赛道，以“头部超配 + 尾部排雷”获取稳健超额", "PART 01  |  基金定位与市场逻辑")

    s.add_kpi_card(52, 108, 278, 102, "产品类型与模拟规模", "1,000 万元", "机构指增定位 · 追求相对稳健增值", C_NAVY)
    s.add_kpi_card(350, 108, 278, 102, "投资范围", "中证 500 成分股", "562期周度 PIT 快照 · 1,323只历史股", C_STEEL)
    s.add_kpi_card(648, 108, 278, 102, "业绩比较基准", "H00905 全收益", "含分红再投资 · 常年约 99% 权益仓位", C_CRIMSON)
    s.add_kpi_card(946, 108, 282, 102, "超额收益双引擎", "精选 + 排雷", "超配优质领跑股 · 清零尾部高危股", C_GREEN)

    # Left card: Why CSI 500 comparison
    s.add_card(52, 232, 574, 434, fill=C_WHITE, border=C_BORDER, top_accent=C_NAVY, shadow=True)
    s.add_text(74, 250, 530, 28, "为什么选择中证 500 指数增强？", size=14.0, color=C_NAVY, bold=True)
    pts = [
        ("对比沪深 300：个股分化更大、选股空间更广",
         "沪深 300 集中于银行白酒等大盘权重，机构覆盖极深；中证 500 汇聚电子、通信、医药、高端制造等细分龙头，横截面收益离散度显著更高。"),
        ("对比中证 1000 / 微盘：流动性充沛、抗踩踏更强",
         "微盘股易受流动性冲击重创（如 2024 年初微盘危机）；中证 500 日均成交活跃，可从容承载千万至亿元级资金双周换仓。"),
        ("契合公募指增生态：Beta 清晰、长期复利可期",
         "高仓位锚定中证 500 全收益指数，既不错失中国新质生产力中盘成长 Beta，又可通过纪律化选股持续累积 Alpha。"),
    ]
    py = 292
    for pt_t, pt_b in pts:
        s.add_rect(74, py + 4, 6, 18, fill=C_STEEL)
        s.add_text(88, py, 518, 24, pt_t, size=11.8, color=C_INK, bold=True)
        s.add_text(88, py + 28, 518, 76, pt_b, size=10.4, color=C_MUTED, line_spacing=1.35)
        py += 118

    # Right card: Dual engine visual explanation
    s.add_card(650, 232, 578, 434, fill=C_WHITE, border=C_BORDER, top_accent=C_CRIMSON, shadow=True)
    s.add_text(672, 250, 534, 28, "纯多头指增如何赚取超额收益？", size=14.0, color=C_CRIMSON, bold=True)

    s.add_card(672, 294, 534, 134, fill=C_BG_GREEN, border=C_GREEN, left_accent=C_GREEN, shadow=False)
    s.add_text(692, 308, 496, 26, "第一引擎：超配头部优质标的（贡献约 36% 毛超额）", size=12.0, color=C_GREEN, bold=True)
    s.add_text(692, 338, 496, 80, "在基准权重之上，对盈利改善、估值合理且短期未过热的高分股给予最高 2.5 倍超配，放大行业内优质龙头的复利贡献。", size=10.6, color=C_INK, line_spacing=1.36)

    s.add_card(672, 444, 534, 134, fill=C_BG_CRIMSON, border=C_CRIMSON, left_accent=C_CRIMSON, shadow=False)
    s.add_text(692, 458, 496, 26, "第二引擎：低配与清零尾部弱势股（贡献约 64% 毛超额）", size=12.0, color=C_CRIMSON, bold=True)
    s.add_text(692, 488, 496, 80, "A 股融券做空受限，但将高估值泡沫、持续亏损或暴涨拥挤的尾部约 80 只差股权重直接归零，即可有效规避年化 -13.2% 的尾部跌幅。", size=10.6, color=C_INK, line_spacing=1.36)

    s.add_text(672, 596, 534, 52, "核心投资哲学：不追求孤立预测个股涨跌点数，专注识别同一交易日内 500 只成分股的“相对强弱排序”。", size=10.8, color=C_NAVY, bold=True, line_spacing=1.35)
    slides.append(s)

    # =========================================================================
    # 05. 因子金融逻辑与引入机器学习的价值
    # =========================================================================
    s = SlideSpec(5, total)
    s.add_header_footer(1, "严选量价、估值与盈利质量纯 PIT 因子，用机器学习刻画非线性交互", "PART 01  |  因子金融逻辑与方法选型")

    pillars = [
        ("短期量价行为（20日×6项）", C_NAVY, C_BG_SOFT,
         "日收益率 · 隔夜跳空 · 日内振幅 · 收盘相对高低位置 · 对数换手率 · 相对均量比",
         "金融逻辑：A 股个人投资者交易活跃，短期急涨伴随巨量高换手常反映情绪过热与筹码拥挤；缩量企稳或适度回调往往具备均值回归潜力。"),
        ("估值与安全边际（纯 PIT）", C_CRIMSON, C_BG_CRIMSON,
         "盈利收益率 EP_TTM (1/PE) · 账面市值比 BP_MRQ (1/PB) · 对数流通市值",
         "金融逻辑：直接采用每日盘后公告实时更新的估值指标（彻底规避免费季报事后重述漏洞），衡量股价下方的盈利与净资产安全垫。"),
        ("盈利质量与换手稳定性", C_GREEN, C_BG_GREEN,
         "估值反推隐含 ROE (PB/PE) · 60日 ΔROE · 60日动量与波动率 · 换手变异系数",
         "金融逻辑：区分“真低估景气改善”与“基本面恶化的价值陷阱”，并引入换手稳定性（量稳逻辑）过滤游资短炒标的。"),
    ]
    for i, (pt, pc, pbg, p1, p2) in enumerate(pillars):
        px = 52 + i * 398
        s.add_card(px, 108, 380, 250, fill=C_WHITE, border=C_BORDER, top_accent=pc, shadow=True)
        s.add_text(px + 18, 124, 344, 26, pt, size=12.5, color=pc, bold=True)
        s.add_rect(px + 18, 156, 344, 54, fill=pbg, rounded=True, radius_px=6)
        s.add_text(px + 26, 162, 328, 44, p1, size=9.8, color=C_INK, bold=True, line_spacing=1.28)
        s.add_text(px + 18, 220, 344, 126, p2, size=10.2, color=C_MUTED, line_spacing=1.36)

    s.add_card(52, 378, 1176, 290, fill=C_WHITE, border=C_BORDER, shadow=True)
    s.add_text(72, 392, 800, 26, "为什么引入轻量级机器学习？（对比传统线性多因子打分的三大投资增益）", size=13.0, color=C_NAVY, bold=True)
    s.add_table(
        x=72, y=426, w=1136,
        col_widths=[176, 310, 350, 300],
        headers=["核心投资痛点", "传统线性多因子（等权/岭回归）", "本策略方法（轻量时序-截面模型）", "实证验证效果"],
        rows=[
            ["因子条件交互", "假设因子独立线性相加，易掉入低估亏损陷阱", "自动学习“低估值 + 盈利改善 + 缩量回调”联合形态", "有效剔除高估过热与亏损伪成长"],
            ["20日走势路径", "仅看区间均值，无法区分先涨后跌与温和走强", "按时间顺序读取 20 日量价演变节奏", "避免人工堆砌数十个技术指标过拟合"],
            ["同日相对强弱", "单股孤立回归，易受市场整体涨跌噪音干扰", "同日 500 只成分股横向对比 + 无偏排序目标", "样本外 RankIC 由 +4.95% 提升至 +7.63%"],
        ],
        row_h=64.0, header_h=38.0, font_size=10.2,
    )
    slides.append(s)

    # =========================================================================
    # 06. SECTION 02 DIVIDER
    # =========================================================================
    slides.append(_make_section_slide(
        6, total, "02", "内容比重 25%",
        "策略核心流程：选股·择时·配置·风控",
        "围绕真实投资问题设计四大协同模块，让每一份预测信号在严守交易规则与低换手约束下转化为可执行组合。",
        [
            ("选股与适应", "横向相对排序", "融合 20 日量价走势与 8 项静态基本面，双周调仓 + 年度滚动更新"),
            ("组合配置", "Soft-Tilt 软倾斜", "锚定流通市值底仓，连续加减配并清零尾部风险股（附三股实案）"),
            ("实盘风控", "合规与诚实审视", "防未来函数、涨跌停锁仓、主板/科创板整手规则，并坦诚剖析风控短板"),
        ],
    ))

    # =========================================================================
    # 07. 策略核心流程总览大图
    # =========================================================================
    s = SlideSpec(7, total)
    s.add_header_footer(2, "从纯 PIT 市场信息到合规组合执行：四大业务模块紧密协同", "PART 02  |  策略端到端核心流程总览")

    s.add_card(52, 106, 1176, 424, fill=C_WHITE, border=C_BORDER, shadow=True)
    s.add_image(66, 118, 1148, 400, FIG_DIR / "00_strategy_architecture_flow.png")

    notes = [
        ("信号预测与组合管理并重", C_NAVY,
         "高 RankIC 只是起点：若直接买前 50 只（Top-50），回撤高达 -22.9% 且换手偏高；必须搭配 Soft-Tilt 宽基锚定与 EMA 平滑，才能实现稳健增强。"),
        ("全链路零未来函数硬约束", C_CRIMSON,
         "历史成分股周度 PIT、基本面由公告日估值实时计算、滚动训练强制留出 10 交易日隔离期、周五收盘信号下周一开盘成交。"),
        ("轻量化与透明可解释", C_GREEN,
         "底层模型仅约 1.5 万参数（普通笔记本 CPU 即可复现），门控权重与选股得分全程可追踪，杜绝不可控黑盒。"),
    ]
    for i, (nt, nc, nb) in enumerate(notes):
        nx = 52 + i * 398
        s.add_card(nx, 546, 380, 124, fill=C_WHITE, border=C_BORDER, left_accent=nc, shadow=True)
        s.add_text(nx + 18, 556, 348, 24, nt, size=11.5, color=nc, bold=True)
        s.add_text(nx + 18, 584, 348, 78, nb, size=9.8, color=C_MUTED, line_spacing=1.32)
    slides.append(s)

    # =========================================================================
    # 08. 选股与择时机制（配门控动态曲线大图）
    # =========================================================================
    s = SlideSpec(8, total)
    s.add_header_footer(2, "选股专注同日 500 股相对排序；择时放弃大盘押注、聚焦风格自适应", "PART 02  |  选股打分与择时适应机制")

    # Left: Intuitive 3-step selection logic
    s.add_card(52, 108, 580, 560, fill=C_WHITE, border=C_BORDER, top_accent=C_NAVY, shadow=True)
    s.add_text(74, 124, 536, 28, "选股模块：TS-CSNet 打分模型的三步金融直觉", size=13.5, color=C_NAVY, bold=True)
    steps = [
        ("第一步：读懂 20 日量价走势形态", C_STEEL, C_BG_SOFT,
         "用轻量时序模块消化过去 20 天涨跌节奏、日内上下影线与换手变化，自动识别“温和蓄势”与“爆量见顶”等走势形态。"),
        ("第二步：动态权衡短期量价与静态基本面", C_CRIMSON, C_BG_CRIMSON,
         "通过可观测的门控权重 g ∈ (0,1) 自动融合量价与估值/盈利信息——不把策略绑死在单一风格上（见右下方实证曲线）。"),
        ("第三步：全市场 500 股横向比强弱 + 排序优化", C_GREEN, C_BG_GREEN,
         "把同日 500 只股票放在一起横向对比，并用无偏相关性与成对排序目标替代传统 MSE，专注排准前后名次而非猜具体涨幅。"),
    ]
    sy = 168
    for st, sc, sbg, sb in steps:
        s.add_card(74, sy, 536, 148, fill=sbg, border=C_BORDER, left_accent=sc, shadow=False)
        s.add_text(92, sy + 14, 500, 26, st, size=11.8, color=sc, bold=True)
        s.add_text(92, sy + 46, 500, 92, sb, size=10.4, color=C_INK, line_spacing=1.35)
        sy += 164

    # Right top: Timing rationale
    s.add_card(652, 108, 576, 290, fill=C_WHITE, border=C_BORDER, top_accent=C_CRIMSON, shadow=True)
    s.add_text(674, 124, 532, 28, "择时模块：不做大盘仓位择时，坚持双周滚动与平滑", size=13.5, color=C_CRIMSON, bold=True)
    s.add_text(674, 162, 532, 224,
               "• 为什么不主动增减股票总仓位？\n"
               "  指增基金职责是稳定提供中盘权益 Beta 并叠加选股超额。A 股牛市涨幅往往集中在极少数交易周，主观择时减仓极易踏空跑输基准，因此常年保持约 99% 权益仓位。\n\n"
               "• 双周调仓 + 因果信号平滑 + 年度滚动更新：\n"
               "  每 10 个交易日调仓一次，相邻两期得分各取 50% 平滑过滤噪音；每年末用最近 3 年样本滚动重训，顺应中长期市场定价迁移。",
               size=10.4, color=C_INK, line_spacing=1.36)

    # Right bottom: Gate weight chart card
    s.add_chart_card(652, 412, 576, 256, "隐式风格适应：门控权重动态调节（均值 g = 0.48）", SLIDE_FIG_DIR / "c17_gate.png",
                     "轮动快年份自动提高量价权重(>0.50)，熊市与估值重构期转向基本面(<0.45)", accent=C_NAVY)
    slides.append(s)

    # =========================================================================
    # 09. Soft-Tilt 软倾斜配置机制与三股真实案例
    # =========================================================================
    s = SlideSpec(9, total)
    s.add_header_footer(2, "Soft-Tilt 软倾斜配置：锚定基准市值分布，连续完成超配、低配与清零", "PART 02  |  投资组合配置方法与真实个股案例")

    # Top visual weight-mapping zones
    zones = [
        ("尾部排雷区（得分 z ≤ -1.0σ）", "倾斜倍数 = 0.0×（直接清零）", "约 80 只高危股权重归零，规避暴跌尾部", C_CRIMSON, C_BG_CRIMSON),
        ("温和低配区（-1.0σ < z < 0）", "倾斜倍数 0.0× ~ 1.0×", "保留部分市值底仓控制跟踪误差，适度减配", C_STEEL, C_BG_SOFT),
        ("精选超配区（得分 z > 0）", "倾斜倍数 1.0× ~ 2.5×（封顶）", "按得分线性加仓优质龙头，单股最高 2.5 倍防集中", C_GREEN, C_BG_GREEN),
    ]
    for i, (zt, zv, zs, zc, zbg) in enumerate(zones):
        zx = 52 + i * 398
        s.add_card(zx, 108, 380, 146, fill=zbg, border=zc, top_accent=zc, shadow=True)
        s.add_text(zx + 18, 122, 344, 24, zt, size=11.5, color=zc, bold=True)
        s.add_text(zx + 18, 152, 344, 30, zv, size=14.0, color=C_INK, bold=True)
        s.add_text(zx + 18, 190, 344, 52, zs, size=10.0, color=C_MUTED, line_spacing=1.32)

    s.add_card(52, 274, 1176, 394, fill=C_WHITE, border=C_BORDER, shadow=True)
    s.add_text(74, 290, 1130, 28, "真实个股权重调整案例（摘自 2026-11-24 系统实盘建仓计划，初始资金 1,000 万元）", size=13.2, color=C_NAVY, bold=True)
    s.add_table(
        x=74, y=328, w=1132,
        col_widths=[104, 168, 96, 124, 92, 168, 96, 284],
        headers=["调整方向", "股票（申万行业）", "基准底仓", "模型排名 (得分)", "倾斜倍数", "计划权重 (整手股数)", "主动偏离", "核心金融依据"],
        rows=[
            ["典型超配", "长飞光纤 (通信)", "0.84%", "#7 (+1.45σ)", "2.45×", "1.87% (900股)", "+1.03%", "ROE 34.7%、ΔROE +13.6%，20日深调 -28% 具强修复赔率"],
            ["典型低配", "康龙化成 (医药生物)", "0.58%", "#422 (-0.82σ)", "0.18×", "0.08% (200股)", "-0.50%", "PE 42.8x 偏高、盈利零增长，近期反弹 +10.6% 后减配"],
            ["清零排雷", "华虹宏力 (电子·科创)", "0.84%", "#500 (-5.14σ)", "0.00×", "0.00% (0股)", "-0.84%", "PE 309x、ROE 仅 2.8%，20日暴涨 +40% 且高换手 6.7%"],
        ],
        row_h=74.0, header_h=40.0, font_size=10.2,
        highlight_rows={0: C_BG_GREEN, 2: C_BG_CRIMSON},
        col_aligns=["center", "left", "center", "center", "center", "center", "center", "left"],
    )
    s.add_text(74, 608, 1132, 46, "直观对比窄幅 Top-50 选股：Soft-Tilt 不仅让年化净超额由 +0.71% 翻倍至 +1.66%，还将相对最大回撤由 -22.94% 压缩至 -14.62%，换手率由 4.11x 降至 2.69x。", size=10.5, color=C_NAVY, bold=True)
    slides.append(s)

    # =========================================================================
    # 10. 风险控制体系与当前不足（诚实审视）
    # =========================================================================
    s = SlideSpec(10, total)
    s.add_header_footer(2, "严守四道实盘合规与成本防线，同时坦诚审视当前风控短板", "PART 02  |  风险控制体系与改进空间")

    # Left 2x2 grid of implemented risk controls
    rc_cards = [
        ("01  防未来函数与泄露纪律", C_NAVY,
         "• 562 期周度 PIT 成分股快照消除幸存者偏差\n"
         "• 公告日估值实时反推基本面，零季报重述漏洞\n"
         "• 滚动切分强制 10 日隔离期（4/4 单元测试验证）"),
        ("02  A 股真实交易限制约束", C_STEEL,
         "• 周五收盘出信号，下周一开盘价真实撮合\n"
         "• 停牌、ST 及开盘涨停股严禁买入\n"
         "• 持仓股遇跌停或停牌自动锁仓至恢复交易"),
        ("03  板块整手与余钱补手执行", C_GREEN,
         "• 主板/创业板严格 100 股整数倍（1手）申报\n"
         "• 科创板（688）严格执行 ≥200 股最低申报门槛\n"
         "• 二阶段贪心补手将现金拖累由 7.36% 压至 0.50%"),
        ("04  持仓分散与换手成本控制", C_GOLD,
         "• 期均持股 416 只，有效持股 270.5 只，前十大仅 8.1%\n"
         "• 双周调仓 + 因果平滑使年化单边换手仅 2.69x\n"
         "• 全程计入分段印花税/佣金/滑点，严禁任何杠杆"),
    ]
    rc_pos = [(52, 108), (434, 108), (52, 396), (434, 396)]
    for (rt, rc, rb), (rx, ry) in zip(rc_cards, rc_pos):
        s.add_card(rx, ry, 366, 268, fill=C_WHITE, border=C_BORDER, top_accent=rc, shadow=True)
        s.add_text(rx + 18, ry + 18, 330, 28, rt, size=12.2, color=rc, bold=True)
        s.add_rect(rx + 18, ry + 52, 330, 1, fill=C_RULE)
        s.add_text(rx + 18, ry + 66, 330, 186, rb, size=10.3, color=C_INK, line_spacing=1.42)

    # Right column: Honest limitations
    s.add_card(820, 108, 408, 556, fill=C_BG_CRIMSON, border=C_CRIMSON, top_accent=C_CRIMSON, shadow=True)
    s.add_text(842, 126, 364, 30, "当前风控的明确不足（诚实反思）", size=14.0, color=C_CRIMSON, bold=True)
    s.add_rect(842, 162, 364, 1, fill="F2C6C9")

    s.add_card(842, 182, 364, 168, fill=C_WHITE, border="F2C6C9", left_accent=C_CRIMSON, shadow=False)
    s.add_text(860, 194, 332, 26, "短板一：缺乏行业中性硬约束", size=11.8, color=C_CRIMSON, bold=True)
    s.add_text(860, 224, 332, 114, "当前仅锚定全市场市值权重，未限制单一行业偏离上限。历史回测对高波动电子行业平均低配 -2.80%，在 2025 科技牛市中造成 -4.22% 的行业配置拖累。", size=10.1, color=C_INK, line_spacing=1.36)

    s.add_card(842, 366, 364, 168, fill=C_WHITE, border="F2C6C9", left_accent=C_CRIMSON, shadow=False)
    s.add_text(860, 378, 332, 26, "短板二：缺乏组合 Beta 下限约束", size=11.8, color=C_CRIMSON, bold=True)
    s.add_text(860, 408, 332, 114, "模型自然偏好低波低换手标的（全周期 Beta=0.838，2025 年降至约 0.80），在熊市抗跌极佳，但在单边暴涨牛市中会因 Beta 折价跑输基准。", size=10.1, color=C_INK, line_spacing=1.36)

    s.add_text(842, 552, 364, 96, "结项改进方向：\n在保持底层选股打分不变下，引入【申万行业内软倾斜标准化】与【组合 Beta ≥ 0.92】轻约束。", size=10.5, color=C_NAVY, bold=True, line_spacing=1.38)
    slides.append(s)

    # =========================================================================
    # 11. SECTION 03 DIVIDER (40% CORE FOCUS)
    # =========================================================================
    slides.append(_make_section_slide(
        11, total, "03", "内容比重 40% · 汇报核心重点",
        "回测、多维归因与适用性分析",
        "不仅展示策略取得了怎样的 9 年样本外成绩，更穿透拆解收益从何而来、2025 年为何跑输以及策略的真实适用边界。",
        [
            ("长期实证绩效", "RankIC +7.63%", "年化 ICIR +2.37（胜率 70.5%），扣费年化净超额 +1.66%、换手 2.69x"),
            ("穿透式业绩归因", "选股 +2.36%/年", "Brinson 行业内选股连胜 8 年；清零尾部 83 只差股贡献 59% 毛超额"),
            ("严谨反思与边界", "不回避负面年份", "深度定量剖析 2025 年正 RankIC 却跑输原因，界定牛熊波动与容量边界"),
        ],
    ))

    # =========================================================================
    # 12. 回测设计与 9 年样本外全指标对比表
    # =========================================================================
    s = SlideSpec(12, total)
    s.add_header_footer(3, "回测设计严谨公平：本策略在预测力、净超额与低换手上全面领先基线", "PART 03  |  回测规范与核心绩效总览")

    badges = [
        ("样本外区间", "2017–2025 完整 9 年（230 期双周调仓，2026+ 严格隔离）", C_NAVY),
        ("滚动防泄露", "每年用前 3 年样本重训 + 24 周验证早停 + 10 日精确隔离期", C_STEEL),
        ("真实成本假设", "买单边 10bp；卖单边 20bp（2023-08印花税减半前）/ 15bp（后）", C_CRIMSON),
        ("公平对照原则", "5 组模型共享同等范围特征信息、同一撮合引擎与同一软倾斜规则", C_GREEN),
    ]
    for i, (bt, bs, bc) in enumerate(badges):
        bx = 52 + i * 298
        s.add_card(bx, 106, 284, 96, fill=C_WHITE, border=C_BORDER, left_accent=bc, shadow=True)
        s.add_text(bx + 16, 116, 256, 22, bt, size=10.5, color=bc, bold=True)
        s.add_text(bx + 16, 142, 256, 52, bs, size=9.2, color=C_MUTED, line_spacing=1.30)

    s.add_card(52, 218, 1176, 452, fill=C_WHITE, border=C_BORDER, shadow=True)
    s.add_text(74, 232, 960, 28, "2017–2025 严格样本外全指标对比表（基准：中证 500 全收益指数 H00905）", size=13.2, color=C_NAVY, bold=True)
    s.add_table(
        x=74, y=268, w=1132,
        col_widths=[216, 98, 98, 98, 102, 112, 86, 108, 108, 106],
        headers=["策略 / 对照组", "双周RankIC", "年化ICIR", "IC胜率", "年化收益", "年化净超额", "信息比IR", "最大回撤", "相对回撤", "单边换手"],
        rows=[
            ["B0 中证500全收益基准", "—", "—", "—", "+3.25%", "0.00%", "0.00", "-39.63%", "0.00%", "0.00x"],
            ["B1 成分股等权组合", "—", "—", "—", "+3.33%", "+0.08%", "-0.02", "-41.39%", "-11.80%", "0.83x"],
            ["M1 线性岭回归基线", "+4.95%", "+1.73", "65.2%", "+4.18%", "+0.91%", "+0.11", "-41.80%", "-10.18%", "3.01x"],
            ["M2 基础神经网络 (MLP)", "+5.88%", "+1.97", "67.4%", "+4.76%", "+1.47%", "+0.19", "-41.11%", "-12.84%", "3.28x"],
            ["M3 时序网络 (GRU+MSE)", "+5.70%", "+1.85", "65.2%", "+4.11%", "+0.83%", "+0.07", "-44.08%", "-13.04%", "2.91x"],
            ["M4 时序+无偏排序损失", "+7.48%", "+2.27", "68.7%", "+4.75%", "+1.46%", "+0.15", "-45.44%", "-15.28%", "2.80x"],
            ["M5 本策略（完整融合增强）", "+7.63%", "+2.37", "70.5%", "+4.95%", "+1.66%", "+0.19", "-42.85%", "-14.62%", "2.69x"],
        ],
        row_h=47.0, header_h=40.0, font_size=10.0,
        highlight_rows={6: C_BG_GREEN},
        col_aligns=["left", "center", "center", "center", "center", "center", "center", "center", "center", "center"],
    )
    slides.append(s)

    # =========================================================================
    # 13. 累计净值走势大图 (c01_nav.png)
    # =========================================================================
    s = SlideSpec(13, total)
    s.add_header_footer(3, "累计净值走势：9 年扣费净值达 1.55（基准 1.33），长期复利稳健领先", "PART 03  |  绩效分析（一）：累计净值曲线")

    s.add_chart_card(52, 108, 804, 560, "2017–2025 样本外累计净值走势（起始 = 1.00，已扣真实双边税费）",
                     SLIDE_FIG_DIR / "c01_nav.png", "粉红底纹标注 2025 年高 Beta 科技单边牛市区间", accent=C_NAVY)

    s.add_kpi_card(876, 108, 352, 114, "本策略期末累计净值", "1.55  (+54.7%)", "中证500全收益基准 1.33 (+33.4%)", C_NAVY)
    s.add_kpi_card(876, 238, 352, 114, "扣费后年化净收益率", "+4.95%", "较线性多因子基线 (+4.18%) 提升 +77bp", C_GREEN)

    s.add_card(876, 368, 352, 300, fill=C_WHITE, border=C_BORDER, left_accent=C_STEEL, shadow=True)
    s.add_text(896, 384, 316, 26, "净值走势三大关键特征", size=12.2, color=C_NAVY, bold=True)
    s.add_text(896, 418, 316, 236,
               "1. 熊市缓冲显著：\n2018 年去杠杆与 2022 年下跌市中，净值回撤幅度明显浅于基准与等权指数。\n\n"
               "2. 震荡期持续创出相对新高：\n2021–2024 年宽幅震荡期内，本策略显著拉开与线性模型 (1.45) 及基准 (1.33) 的差距。\n\n"
               "3. 高仓位紧跟权益复利：\n2025 年虽跑输指数弹性，但绝对净值仍实现 +18.9% 正收益并创历史新高 1.55。",
               size=10.0, color=C_INK, line_spacing=1.34)
    slides.append(s)

    # =========================================================================
    # 14. 相对超额净值曲线大图 (c02_excess.png)
    # =========================================================================
    s = SlideSpec(14, total)
    s.add_header_footer(3, "相对超额曲线：前 8 年稳步攀升至峰值 1.34，2025 单边牛市回撤收敛至 1.16", "PART 03  |  绩效分析（二）：相对基准超额轨迹")

    s.add_chart_card(52, 108, 804, 560, "相对中证 500 全收益指数累计超额净值（策略净值 / 基准净值）",
                     SLIDE_FIG_DIR / "c02_excess.png", "斜率向上代表跑赢基准，向下代表跑输基准", accent=C_GREEN)

    s.add_kpi_card(876, 108, 352, 114, "2017–2024 峰值相对净值", "1.34  (+34.0%)", "2024年8月创超额高点 · 8年稳步复利", C_GREEN)
    s.add_kpi_card(876, 238, 352, 114, "2025 单年相对回撤后终值", "1.16  (+16.0%)", "全周期年化净超额 +1.66% · 换手 2.69x", C_CRIMSON)

    s.add_card(876, 368, 352, 300, fill=C_WHITE, border=C_BORDER, left_accent=C_CRIMSON, shadow=True)
    s.add_text(896, 384, 316, 26, "超额轨迹的客观解读", size=12.2, color=C_CRIMSON, bold=True)
    s.add_text(896, 418, 316, 236,
               "• 为什么等权基准 (虚线) 几乎无超额？\n扣除真实换手税费后，简单等权 9 年相对净值仅 1.01，证明超额并非来自小盘等权红利。\n\n"
               "• M4 / M5 在 2022–2024 明显领跑：\n无偏排序与截面比较机制在震荡分化市中展现出远超线性模型的选股弹性。\n\n"
               "• 不回避 2025 回撤：\n所有低波/价值倾向模型在 2025 同步回撤，提示组合层风格风控的必要性。",
               size=10.0, color=C_INK, line_spacing=1.34)
    slides.append(s)

    # =========================================================================
    # 15. 分年度超额表现全宽大图 (c03_yearly.png)
    # =========================================================================
    s = SlideSpec(15, total)
    s.add_header_footer(3, "分年度表现：熊市与震荡年大幅跑赢，2024 微盘危机稳健获正超额 +4.3%", "PART 03  |  绩效分析（三）：2017–2025 逐年拆解")

    s.add_chart_card(52, 106, 1176, 422, "2017–2025 逐年相对中证 500 全收益净超额对比（横轴附当年基准涨跌幅）",
                     SLIDE_FIG_DIR / "c03_yearly.png", "深蓝：本策略跑赢年份  |  深红：本策略跑输年份  |  浅蓝：线性多因子对照 (M1)", accent=C_NAVY)

    yr_take = [
        ("下跌熊市保护极强", C_GREEN,
         "2018 年（基准 -32.5%）斩获 +8.2% 超额；2022 年（基准 -18.9%）斩获 +7.6% 超额，显著减轻投资者熊市痛感。"),
        ("抗微盘流动性冲击", C_NAVY,
         "2023–2024 年连续跑赢（+3.7%、+4.3%），在 2024 年 2 月量化微盘股大回撤期间平稳渡劫，大幅优于线性基线 (+1.0%)。"),
        ("急涨牛市弹性不足", C_CRIMSON,
         "跑输主要发生在 2020 (+22.3%) 与 2025 (+32.5%) 两年高估值成长急涨牛市，清晰指向组合低 Beta 防御暴露特征。"),
    ]
    for i, (yt, yc, yb) in enumerate(yr_take):
        yx = 52 + i * 398
        s.add_card(yx, 544, 380, 124, fill=C_WHITE, border=C_BORDER, top_accent=yc, shadow=True)
        s.add_text(yx + 18, 558, 344, 24, yt, size=11.5, color=yc, bold=True)
        s.add_text(yx + 18, 586, 344, 74, yb, size=9.8, color=C_MUTED, line_spacing=1.32)
    slides.append(s)

    # =========================================================================
    # 16. 从0到1迭代优化与递进消融双大图 (c04 & c05)
    # =========================================================================
    s = SlideSpec(16, total)
    s.add_header_footer(3, "从 0 到 1 四轮迭代：攻克“指数排序波动率陷阱”，实现预测力与净超额双登顶", "PART 03  |  研究迭代过程与模块消融证据")

    # Top 4 compact horizontal iteration cards
    it_steps = [
        ("v0 线性/MLP 基线", "RankIC +4.95%~+5.88%", "难以刻画 20 日序列节奏", C_STEEL, C_WHITE),
        ("v1 指数 ListNet 踩坑", "RankIC 跌至 -4.63%！", "异方差放大高波彩票股偏好", C_RED, C_BG_CRIMSON),
        ("v2 无偏排序 + 横向对比", "RankIC 跃升 +7.63%", "消除波动率偏差，9/9年验证全正", C_NAVY, C_BG_SOFT),
        ("v3 Soft-Tilt 配置升级", "净超额翻倍至 +1.66%", "较 Top-50 (+0.71%) 大降回撤换手", C_GREEN, C_BG_GREEN),
    ]
    for i, (it, iv, isub, ic, ibg) in enumerate(it_steps):
        ix = 52 + i * 298
        s.add_card(ix, 104, 284, 108, fill=ibg, border=ic if i == 1 else C_BORDER, top_accent=ic, shadow=True)
        s.add_text(ix + 14, 116, 256, 22, it, size=10.5, color=ic, bold=True)
        s.add_text(ix + 14, 140, 256, 28, iv, size=13.5, color=C_INK, bold=True)
        s.add_text(ix + 14, 172, 256, 32, isub, size=9.2, color=C_MUTED)

    s.add_chart_card(52, 228, 576, 440, "消融证据（一）：选股预测力 RankIC 与年化 ICIR",
                     SLIDE_FIG_DIR / "c04_ablation_ic.png", "M3 → M4 替换为无偏排序目标贡献最大单步跃升 (+1.78pp)", accent=C_NAVY)
    s.add_chart_card(652, 228, 576, 440, "消融证据（二）：扣减真实税费后的年化净超额与 IR",
                     SLIDE_FIG_DIR / "c05_ablation_excess.png", "M5 完整融合策略实现最高净超额 (+1.66%) 与最低换手 (2.69x)", accent=C_GREEN)
    slides.append(s)

    # =========================================================================
    # 17. 持仓特征与十分位/多空排雷归因 (c07_decile + c10_donut)
    # =========================================================================
    s = SlideSpec(17, total)
    s.add_header_footer(3, "持仓分散且分层严格单调：清零尾部 83 只高危股贡献了 59% 的毛超额", "PART 03  |  持仓分析与多空排雷归因")

    # Top 4 holding profile KPI cards
    s.add_kpi_card(52, 104, 278, 98, "期均持股 / 自动清零", "416 只 / 83 只", "有效持股数 270.5 只 · 宽基分散", C_NAVY)
    s.add_kpi_card(350, 104, 278, 98, "权重集中度", "前十大 8.12%", "单股最大均值仅 1.00% · 无重仓押注", C_STEEL)
    s.add_kpi_card(648, 104, 278, 98, "主动管理份额 (Active Share)", "30.1%", "温和偏离基准 · 兼顾跟踪与增强", C_GREEN)
    s.add_kpi_card(946, 104, 282, 98, "尾部排雷收益占比", "59.3% (+1.21%)", "避开 D10 暴跌股是纯多头关键胜负手", C_CRIMSON)

    s.add_chart_card(52, 218, 676, 450, "模型得分十分位年化相对超额（D1 最高 → D10 最低）",
                     SLIDE_FIG_DIR / "c07_decile.png", "前六组全部正超额，第 10 组（尾部最差 50 只）年化跑输高达 -13.2%", accent=C_GREEN)
    s.add_chart_card(748, 218, 480, 450, "年化毛超额 (+2.04%) 来源分解",
                     SLIDE_FIG_DIR / "c10_donut.png", "超配头部占 36%，低配与清零尾部共占 64%", accent=C_CRIMSON)
    slides.append(s)

    # =========================================================================
    # 18. Brinson 行业配置与行业内选股归因双大图 (c08 & c09)
    # =========================================================================
    s = SlideSpec(18, total)
    s.add_header_footer(3, "Brinson 归因：核心收益来自“行业内选股”（+2.36%/年，前 8 年全正）而非赌赛道", "PART 03  |  归因分析：行业配置 vs 行业内个股选择")

    s.add_chart_card(52, 106, 576, 444, "逐年 Brinson 分解：行业配置效应 vs 行业内选股效应",
                     SLIDE_FIG_DIR / "c08_brinson_yearly.png", "2017–2024 连续 8 年行业内选股（深蓝柱）全部为正贡献", accent=C_NAVY)
    s.add_chart_card(652, 106, 576, 444, "申万一级行业年化超额贡献 Top 6 与 Bottom 5",
                     SLIDE_FIG_DIR / "c09_industry.png", "非银(+0.57%)、医药(+0.45%)居前；电子(-0.53%)因低配拖累最大", accent=C_CRIMSON)

    s.add_card(52, 566, 1176, 104, fill=C_WHITE, border=C_BORDER, left_accent=C_NAVY, shadow=True)
    s.add_text(74, 578, 1132, 80,
               "Brinson 定量汇总（2017–2025 年化）：  行业内个股选择效应 = +2.36%/年（核心正收益来源）   |   行业配置效应 = +0.39%/年   |   交互效应 = -0.70%/年\n"
               "关键结论：策略在绝大多数年份依靠在同一行业内挑出优质标的盈利；2025 年电子与通信大幅领涨时，低配科技赛道导致行业配置单年回撤 -4.22%。",
               size=10.3, color=C_INK, bold=True, line_spacing=1.42)
    slides.append(s)

    # =========================================================================
    # 19. 深度反思（一）：2025 年选股排序能力失效了吗？ (c06_cum_ic)
    # =========================================================================
    s = SlideSpec(19, total)
    s.add_header_footer(3, "深度反思（一）：2025 年累计 RankIC 仍在稳步创新高，排序能力并未失效", "PART 03  |  专题剖析：为什么正 RankIC 未转化为正超额？")

    s.add_chart_card(52, 108, 796, 560, "2017–2025 样本外双周 RankIC 累计曲线（斜率越陡且越平滑，排序越有效）",
                     SLIDE_FIG_DIR / "c06_cum_ic.png", "2025 年阴影期内本策略累计 RankIC 仍保持正向爬升（全年期均 +3.97%）", accent=C_NAVY)

    s.add_card(868, 108, 360, 248, fill=C_BG_CRIMSON, border=C_CRIMSON, top_accent=C_CRIMSON, shadow=True)
    s.add_text(888, 126, 320, 28, "核心实证悖论", size=13.5, color=C_CRIMSON, bold=True)
    s.add_text(888, 162, 320, 178,
               "• 2025 年信号端：\n双周样本外期均 RankIC = +3.97%，累计 IC 曲线继续向上攀升至 17.3（远超线性基线 11.2）。\n\n"
               "• 2025 年组合端：\n策略绝对收益 +18.9%，但相对暴涨 +32.3% 的中证 500 跑输 -10.16%！",
               size=10.4, color=C_INK, line_spacing=1.38)

    s.add_card(868, 374, 360, 294, fill=C_WHITE, border=C_BORDER, top_accent=C_NAVY, shadow=True)
    s.add_text(888, 392, 320, 28, "给量化投资的核心启示", size=13.0, color=C_NAVY, bold=True)
    s.add_text(888, 428, 320, 224,
               "选股打分（RankIC）衡量 500 只股票等权排序准确度；而真实组合表现还取决于：\n\n"
               "1. 组合相对基准的 Beta 敞口\n"
               "2. 行业权重偏离（赛道集中度）\n"
               "3. 指数领涨权重股的风格属性\n\n"
               "→ 下一页用风格暴露数据定量破案！",
               size=10.4, color=C_INK, line_spacing=1.36)
    slides.append(s)

    # =========================================================================
    # 20. 深度反思（二）：2025 年跑输基准的三重定量机制 (c11_style)
    # =========================================================================
    s = SlideSpec(20, total)
    s.add_header_footer(3, "深度反思（二）：低 Beta 防御倾斜、科技低配与市值加权错位共同导致 2025 跑输", "PART 03  |  专题剖析：风格暴露与 CAPM 定量拆解")

    s.add_chart_card(52, 108, 690, 560, "主动风格暴露对比：9 年均值 vs 2025 年极端防御偏移",
                     SLIDE_FIG_DIR / "c11_style.png", "2022–2024 三年熊市训练使 2025 年低波(-0.73)、低换手(-0.64)与价值(+0.45)暴露推至极值", accent=C_CRIMSON)

    reasons_25 = [
        ("机制 1：低 Beta 敞口在急涨牛市中的数学折价", C_CRIMSON, C_BG_CRIMSON,
         "全周期组合 CAPM Beta = 0.838（剥离 Beta 后年化纯 Alpha = +1.97%）。2025 年实时 Beta 降至约 0.80，面对基准 +32.3% 的暴涨，仅 Beta 缺口 (0.80 - 1) × 32.3% ≈ -6.5% 就造成了超六成相对回撤。"),
        ("机制 2：无行业中性约束导致低配高波科技板块", C_NAVY, C_BG_SOFT,
         "高景气 AI/半导体股票普遍伴随高估值、高波动与高换手，被无行业约束的全局打分系统性低配，导致 2025 年行业配置单项拖累 -4.22%。"),
        ("机制 3：等权排序正确 ≠ 市值加权跑赢", C_STEEL, C_WHITE,
         "即使模型在 450 只普通成分股上排序正确（贡献 +3.97% 正 RankIC），只要清零了少数几只翻倍的大市值高估值科技权重股，市值加权指数仍会大幅拉开差距。"),
    ]
    ry = 108
    for rt, rc, rbg, rb in reasons_25:
        s.add_card(762, ry, 466, 176, fill=rbg, border=C_BORDER, left_accent=rc, shadow=True)
        s.add_text(782, ry + 14, 430, 24, rt, size=11.3, color=rc, bold=True)
        s.add_text(782, ry + 44, 430, 120, rb, size=9.9, color=C_INK, line_spacing=1.34)
        ry += 192
    slides.append(s)

    # =========================================================================
    # 21. 适用性分析（一）：市场行情与波动环境 (c12_regime)
    # =========================================================================
    s = SlideSpec(21, total)
    s.add_header_footer(3, "适用性分析（一）：熊市（+8.08% / IR 1.82）与震荡市优势显著，单边急涨牛承压", "PART 03  |  策略优势环境与失效边界")

    s.add_chart_card(52, 108, 796, 560, "不同市场趋势与波动率环境下的年化超额收益与信息比率",
                     SLIDE_FIG_DIR / "c12_regime.png", "左三柱按 60 日指数涨跌幅划分牛/震荡/熊；右两柱按 20 日波动率中位数划分高低波", accent=C_GREEN)

    s.add_card(868, 108, 360, 272, fill=C_BG_GREEN, border=C_GREEN, top_accent=C_GREEN, shadow=True)
    s.add_text(888, 124, 320, 26, "三大优势主战场（合计占 ~80% 时间）", size=12.0, color=C_GREEN, bold=True)
    s.add_text(888, 158, 320, 208,
               "1. 下跌熊市（占 15% 交易日）：\n年化超额 +8.08%，IR 高达 +1.82，日胜率 57.1%，尾部排雷红利充分释放。\n\n"
               "2. 震荡整理市（占 65% 交易日）：\n年化超额 +2.01%（IR +0.39），在 A 股最常见的震荡慢市中稳步复利。\n\n"
               "3. 高波动分化行情：\n年化超额 +2.97%（IR +0.45），横截面分化越大，选股空间越充裕。",
               size=9.9, color=C_INK, line_spacing=1.33)

    s.add_card(868, 396, 360, 272, fill=C_BG_CRIMSON, border=C_CRIMSON, top_accent=C_CRIMSON, shadow=True)
    s.add_text(888, 412, 320, 26, "两类承压边界（明确告知投资人）", size=12.0, color=C_CRIMSON, bold=True)
    s.add_text(888, 446, 320, 208,
               "1. 单边急涨牛市（60日涨 >+8%，占 20%）：\n年化超额 -6.85%（IR -0.84），情绪驱动普涨与高 Beta 领涨阶段易跑输。\n\n"
               "2. 极低波动沉闷市：\n年化超额 -0.78%，个股缺乏离散度时超额较难覆盖交易成本。\n\n"
               "产品配置建议：极适合作为稳健底仓型中盘指增配置。",
               size=9.9, color=C_INK, line_spacing=1.33)
    slides.append(s)

    # =========================================================================
    # 22. 适用性分析（二）：宏观周期、成本敏感性与资金容量 (c13 & c14)
    # =========================================================================
    s = SlideSpec(22, total)
    s.add_header_footer(3, "适用性分析（二）：前三大宏观周期全正超额，单边 30bp 极端成本下仍保持正收益", "PART 03  |  周期稳健性、交易摩擦与资金容量边界")

    s.add_chart_card(52, 106, 576, 438, "四大历史宏观周期：期均 RankIC 与年化净超额对比",
                     SLIDE_FIG_DIR / "c13_era.png", "2017–2024 三个完整周期全部取得正净超额，2022–2024 高达 +5.21%/年", accent=C_NAVY)
    s.add_chart_card(652, 106, 576, 438, "交易摩擦压力测试：0bp 至 30bp 单边成本净超额表现",
                     SLIDE_FIG_DIR / "c14_cost.png", "得益于 2.69x 低换手，即使单边 30bp（双边 60bp）极端冲击下仍获 +0.80% 正超额", accent=C_STEEL)

    s.add_card(52, 560, 1176, 108, fill=C_WHITE, border=C_BORDER, left_accent=C_GREEN, shadow=True)
    s.add_text(74, 572, 1132, 84,
               "资金规模适用边界分析：\n"
               "• 最适规模（1,000 万元 ~ 5 亿元，机构/公募指增产品）：420 只左右宽基持仓单笔成交远低于中证 500 成分股日均成交额 5% 冲击红线，且经余钱补手后现金拖累 ≤0.50%。\n"
               "• 小资金边界（100 万元以下个人账户）：直接买 400 只股票会受制于 100 股整手与科创板 200 股门槛，建议复用同一打分系统切换为精选 Top-30/50 方案。",
               size=10.0, color=C_INK, line_spacing=1.38)
    slides.append(s)

    # =========================================================================
    # 23. SECTION 04 DIVIDER (20% PAPER TRADING)
    # =========================================================================
    slides.append(_make_section_slide(
        23, total, "04", "内容比重 20%",
        "模拟交易与详细持仓配置",
        "基于 2026-11-24 收盘实时市场数据与生产版模型输出，制定 100% 符合 A 股板块交易限制的 1,000 万元模拟建仓计划。",
        [
            ("口径严谨透明", "建仓计划 vs 实盘成交", "严格区分 11-24 盘后冻结的模拟建仓计划与后续逐期记录的实际成交结果"),
            ("100% 交易合规", "主板100股/科创200股", "股票仓位 99.40%（422 只）、预留费用 0.10%、现金储备 0.50%、清零 78 只"),
            ("闭环跟踪复盘", "结项三大核验", "持续跟踪重仓股收益贡献、78 只清零股排雷效果与模拟期实时 Brinson 归因"),
        ],
    ))

    # =========================================================================
    # 24. 2026-11-24 模拟建仓计划总览与行业分布大图 (c15_live_industry)
    # =========================================================================
    s = SlideSpec(24, total)
    s.add_header_footer(4, "2026-11-24 模拟建仓计划：99.4% 满仓运作，滚动重训后自动适应科技与通信主线", "PART 04  |  模拟账户配置总览与行业结构")

    s.add_kpi_card(52, 104, 278, 98, "初始资金 / 基准点位", "1,000.00 万元", "2026-11-24 中证500收盘 6,112.43", C_NAVY)
    s.add_kpi_card(350, 104, 278, 98, "计划股票市值 / 仓位", "994.00 万 (99.4%)", "持仓 422 只 · 清零排雷 78 只", C_GREEN)
    s.add_kpi_card(648, 104, 278, 98, "预估交易费 / 现金储备", "0.99万 / 5.00万", "单边 10bp 计提 · 0.50% 现金无杠杆", C_STEEL)
    s.add_kpi_card(946, 104, 282, 98, "组合集中度", "前十大 10.55%", "单股最高 1.87% · 门控均值 g=0.45", C_CRIMSON)

    s.add_chart_card(52, 218, 780, 450, "模拟建仓计划前 12 大申万行业权重 vs 基准底仓权重 (2026-11-24)",
                     SLIDE_FIG_DIR / "c15_live_industry.png", "右侧数字为相对基准的主动偏离权重", accent=C_NAVY)

    s.add_card(852, 218, 376, 206, fill=C_BG_CRIMSON, border=C_CRIMSON, left_accent=C_CRIMSON, shadow=True)
    s.add_text(872, 232, 340, 24, "严格区分：建仓计划 vs 实际成交", size=11.5, color=C_CRIMSON, bold=True)
    s.add_text(872, 262, 340, 150,
               "• 本表为基于 2026-11-24 收盘参考价测算的【模拟建仓计划】（全量 422 只股数已归档锁定）。\n"
               "• 【实际模拟成交结果】自下一交易日开盘起按真实市场行情、主板100股/科创板≥200股整手规则及双周调仓纪律逐期跟踪记录。",
               size=9.8, color=C_INK, line_spacing=1.34)

    s.add_card(852, 440, 376, 228, fill=C_BG_GREEN, border=C_GREEN, left_accent=C_GREEN, shadow=True)
    s.add_text(872, 454, 340, 24, "滚动重训后的周期自适应亮点", size=11.5, color=C_GREEN, bold=True)
    s.add_text(872, 484, 340, 172,
               "生产版模型纳入 2025 年样本滚动训练后，显著修正了过往低配科技的惯性：\n"
               "• 高景气【通信】板块主动超配 +3.95%；\n"
               "• 第一大行业【电子】配置 17.03%（近标配），在行业内精选 48 只盈利优质股，同时清零 19 只高估亏损股（规避 6.41% 风险）。",
               size=9.8, color=C_INK, line_spacing=1.34)
    slides.append(s)

    # =========================================================================
    # 25. 前15大重仓图 + 超配/低配/剔除真实个股理由对照 (c16_live_top15)
    # =========================================================================
    s = SlideSpec(25, total)
    s.add_header_footer(4, "个股配置理由：超配高盈利回调龙头、低配景气转弱股、清零高估过热泡沫", "PART 04  |  重仓股分布与超配/低配/剔除代表案例")

    s.add_chart_card(52, 106, 560, 562, "前 15 大重仓股计划权重与基准底仓对比",
                     SLIDE_FIG_DIR / "c16_live_top15.png", "红竖线为该股基准底仓权重；全量 422 只清单见附件 CSV", accent=C_NAVY)

    s.add_card(630, 106, 598, 562, fill=C_WHITE, border=C_BORDER, shadow=True)
    s.add_text(648, 120, 562, 26, "三类典型调整标的真实数据解析（2026-11-24 输出）", size=12.5, color=C_NAVY, bold=True)
    s.add_table(
        x=648, y=154, w=562,
        col_widths=[56, 118, 104, 82, 202],
        headers=["类别", "标的 (行业)", "计划(主动)", "排名", "真实指标与系统配置理由"],
        rows=[
            ["超配", "长飞光纤\n601869 通信", "1.87%\n(+1.03%)", "#7", "ROE 34.7%, ΔROE +13.6%, PE 26.6x, 20日回调 -28%, 买 900 股"],
            ["超配", "源杰科技\n688498 电子", "1.80%\n(+0.96%)", "#22", "科创合规 300 股; ROE 31.7%, ΔROE +17.6% 加速, 回调 -28%"],
            ["超配", "德明利\n001309 电子", "0.67%\n(+0.32%)", "#41", "PE 仅 5.7x, 隐含 ROE 78.7%, ΔROE +16.7%, 回调 -25% 买 400 股"],
            ["低配", "康龙化成\n300759 医药", "0.08%\n(-0.50%)", "#422", "PE 42.8x 偏高, ΔROE +0.1% 停滞, 近20日涨 +11% 后压减至 200 股"],
            ["低配", "赤峰黄金\n600988 有色", "0.20%\n(-0.35%)", "#391", "近60日盈利回落 ΔROE -2.7%, PB 5.0x 偏高, 削减六成权重"],
            ["清零", "华虹宏力\n688347 电子", "0.00%\n(-0.84%)", "#500", "PE 309x, ROE 2.8%, 20日暴涨 +40% 换手 6.7%, 高估过热清零"],
            ["清零", "燕东微\n688172 电子", "0.00%\n(-0.39%)", "#499", "亏损 PE -46.7x, ROE -6.1% 且继续恶化, 反抽 +17% 果断清零"],
        ],
        row_h=65.0, header_h=36.0, font_size=8.9,
        highlight_rows={0: C_BG_GREEN, 1: C_BG_GREEN, 2: C_BG_GREEN, 3: "F8FAFD", 4: "F8FAFD", 5: C_BG_CRIMSON, 6: C_BG_CRIMSON},
        col_aligns=["center", "left", "center", "center", "left"],
    )
    slides.append(s)

    # =========================================================================
    # 26. 模拟跟踪方案、结项风控优化与严谨研究边界总结
    # =========================================================================
    s = SlideSpec(26, total)
    s.add_header_footer(4, "总结与展望：严守实证研究边界，扎实推进模拟跟踪与结项风控升级", "PART 04  |  模拟跟踪计划与研究总结")

    cols = [
        ("一、已验证结论（有实证支撑）", C_GREEN, C_BG_GREEN,
         "1. 融合选股体系显著跑赢传统基线\n"
         "9年样本外双周 RankIC 达 +7.63%（ICIR +2.37，胜率 70.5%），扣真实税费年化净超额 +1.66%，换手低至 2.69x。\n\n"
         "2. 超额收益来源清晰透明\n"
         "Brinson 行业内选股（+2.36%/年）连续 8 年为正；清零尾部约 83 只弱势股贡献约 59% 毛超额。\n\n"
         "3. 优势环境与抗摩擦稳健\n"
         "熊市（+8.08%）与震荡市（+2.01%）优势突出，30bp 极端单边摩擦下仍保持正净超额。"),
        ("二、模拟交易跟踪与结项复盘", C_NAVY, C_BG_SOFT,
         "跟踪起点：2026-11-25 开盘起正式记录实盘成交净值，严格按双周周期再平衡。\n\n"
         "结项汇报三大定量复盘任务：\n"
         "• 重仓归因：核验长飞光纤、源杰科技等前 15 大超配股的真实盈亏贡献；\n"
         "• 排雷检验：统计本期被清零的 78 只高危股后续真实涨跌幅，检验排雷有效性；\n"
         "• 偏差审计：对比建仓计划与实际开盘成交滑点、现金占用及行业 Brinson 归因。"),
        ("三、结项优化方向（待验证假设）", C_CRIMSON, C_BG_CRIMSON,
         "针对 2025 年高 Beta 单边牛市跑输（-10.16%）的核心短板，结项阶段聚焦组合风控升级：\n\n"
         "1. 申万行业内软倾斜标准化\n"
         "将超配与清零约束在各一级行业内部完成，从机制上消除单一热门行业大幅低配风险。\n\n"
         "2. 组合 Beta 下限约束（Beta ≥ 0.92）\n"
         "在保留熊市排雷保护力的同时，测试能否显著压缩单边急涨牛市的相对回撤。"),
    ]
    for i, (ct, cc, cbg, cb) in enumerate(cols):
        cx = 52 + i * 398
        s.add_card(cx, 108, 380, 476, fill=cbg, border=C_BORDER, top_accent=cc, shadow=True)
        s.add_text(cx + 18, 124, 344, 28, ct, size=12.5, color=cc, bold=True)
        s.add_rect(cx + 18, 158, 344, 1, fill=C_BORDER)
        s.add_text(cx + 18, 172, 344, 396, cb, size=10.1, color=C_INK, line_spacing=1.38)

    s.add_card(52, 602, 1176, 62, fill=C_NAVY, border=C_NAVY, shadow=True, radius_px=10)
    s.add_text(52, 608, 1176, 50, "感谢老师与各位同学聆听，敬请批评指正！   |   中证500量价与基本面融合增强策略 · 完整报告与 422 只建仓清单已同步归档",
               size=11.5, color=C_WHITE, bold=True, align="center", valign="middle")
    slides.append(s)

    return slides


def _xml_bytes(el: etree._Element) -> bytes:
    return etree.tostring(el, xml_declaration=True, encoding="UTF-8", standalone=True)


def write_pptx(slides: List[SlideSpec], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for sl in slides:
        sl._img.save(PREVIEW_DIR / f"slide_{sl.slide_no:02d}.png")

    media_map = {}
    media_counter = 1
    for sl in slides:
        for _rid, img_p in sl.images:
            if img_p not in media_map:
                media_map[img_p] = f"image{media_counter}.png"
                media_counter += 1

    ns_ct_uri = "http://schemas.openxmlformats.org/package/2006/content-types"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        ct = etree.Element("Types", xmlns=ns_ct_uri)
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

        rels = etree.Element("Relationships", xmlns=NS_REL)
        etree.SubElement(rels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", Target="ppt/presentation.xml")
        zf.writestr("_rels/.rels", _xml_bytes(rels))

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

        zf.writestr("ppt/presProps.xml", b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>')
        zf.writestr("ppt/viewProps.xml", b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr></p:viewPr>')
        zf.writestr("ppt/tableStyles.xml", b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>')

        theme_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="{NS_A}" name="PKU Executive Minimalist">
  <a:themeElements>
    <a:clrScheme name="PKU">
      <a:dk1><a:srgbClr val="{C_INK}"/></a:dk1>
      <a:lt1><a:srgbClr val="{C_WHITE}"/></a:lt1>
      <a:dk2><a:srgbClr val="{C_NAVY}"/></a:dk2>
      <a:lt2><a:srgbClr val="{C_CANVAS}"/></a:lt2>
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

        for img_p, mname in media_map.items():
            zf.write(img_p, f"ppt/media/{mname}")

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
