#!/usr/bin/env python3
"""将 VRAG_创新方案文档.md 转换为排版精良的 PDF 文件"""

import re
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, KeepTogether, HRFlowable)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 注册中文字体 (TrueType only, reportlab 不支持 PostScript outlines) ──
FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/arphic/uming.ttc"  # 用于粗体
FONT_MONO_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

pdfmetrics.registerFont(TTFont('CJK', FONT_PATH))
pdfmetrics.registerFont(TTFont('CJKBold', FONT_BOLD_PATH, subfontIndex=0))
pdfmetrics.registerFont(TTFont('DejaVuMono', FONT_MONO_PATH))

# ── 自定义样式 ────────────────────────────────────────────
TITLE_COLOR = HexColor('#1a365d')
H2_COLOR = HexColor('#2c5282')
H3_COLOR = HexColor('#2b6cb0')
ACCENT_COLOR = HexColor('#3182ce')
TABLE_HEADER_BG = HexColor('#2c5282')
TABLE_ROW_ALT = HexColor('#ebf4ff')
CODE_BG = HexColor('#f7fafc')
BORDER_COLOR = HexColor('#e2e8f0')

styles = getSampleStyleSheet()

style_title = ParagraphStyle('CNTitle', fontName='CJKBold', fontSize=22,
                             leading=30, textColor=TITLE_COLOR, alignment=TA_CENTER,
                             spaceAfter=6*mm)

style_subtitle = ParagraphStyle('CNSubtitle', fontName='CJK', fontSize=11,
                                leading=16, textColor=grey, alignment=TA_CENTER,
                                spaceAfter=10*mm)

style_h1 = ParagraphStyle('CNH1', fontName='CJKBold', fontSize=16,
                          leading=24, textColor=TITLE_COLOR, spaceBefore=10*mm,
                          spaceAfter=4*mm)

style_h2 = ParagraphStyle('CNH2', fontName='CJKBold', fontSize=13,
                          leading=20, textColor=H2_COLOR, spaceBefore=8*mm,
                          spaceAfter=3*mm)

style_h3 = ParagraphStyle('CNH3', fontName='CJKBold', fontSize=11,
                          leading=17, textColor=H3_COLOR, spaceBefore=5*mm,
                          spaceAfter=2*mm)

style_body = ParagraphStyle('CNBody', fontName='CJK', fontSize=10,
                            leading=18, textColor=black, alignment=TA_JUSTIFY,
                            spaceAfter=2*mm, firstLineIndent=0)

style_body_indent = ParagraphStyle('CNBodyIndent', parent=style_body,
                                   firstLineIndent=20)

style_code = ParagraphStyle('CNCode', fontName='DejaVuMono', fontSize=8,
                            leading=12, textColor=HexColor('#2d3748'),
                            backColor=CODE_BG, leftIndent=10, rightIndent=10,
                            spaceBefore=2*mm, spaceAfter=2*mm,
                            borderPadding=6, borderWidth=0.5,
                            borderColor=BORDER_COLOR)

style_bullet = ParagraphStyle('CNBullet', parent=style_body, leftIndent=12,
                              bulletIndent=4, spaceBefore=1*mm, spaceAfter=1*mm,
                              firstLineIndent=0)

style_table_cell = ParagraphStyle('CNTableCell', fontName='CJK', fontSize=8,
                                  leading=13, textColor=black)

style_table_header = ParagraphStyle('CNTableHeader', fontName='CJKBold',
                                    fontSize=8, leading=13, textColor=white)

style_caption = ParagraphStyle('CNCaption', fontName='CJK', fontSize=9,
                               leading=14, textColor=grey, alignment=TA_CENTER,
                               spaceBefore=2*mm, spaceAfter=4*mm)

style_quote = ParagraphStyle('CNQuote', fontName='CJK', fontSize=10,
                             leading=17, textColor=HexColor('#2d3748'),
                             leftIndent=15, rightIndent=15, spaceBefore=3*mm,
                             spaceAfter=3*mm, borderWidth=3, borderColor=ACCENT_COLOR,
                             borderPadding=8, backColor=HexColor('#f0f5ff'))


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR,
                      spaceBefore=3*mm, spaceAfter=3*mm)


def make_table(headers, rows, col_widths=None):
    """创建带样式的表格"""
    header_cells = [Paragraph(h, style_table_header) for h in headers]
    data = [header_cells]
    for row in rows:
        data.append([Paragraph(str(c), style_table_cell) for c in row])

    if col_widths is None:
        avail = A4[0] - 30*mm
        col_widths = [avail / len(headers)] * len(headers)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'CJKBold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, TABLE_ROW_ALT]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def build_pdf(md_path: str, pdf_path: str):
    """解析 Markdown 并生成 PDF"""
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=18*mm, bottomMargin=18*mm,
                            title='VRAG 创新方案文档',
                            author='VRAG Team')

    story = []

    # ── 解析 Markdown ──────────────────────────────────────
    lines = text.split('\n')
    i = 0
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []
    table_header = []
    in_quote = False
    quote_lines = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            code_text = '\n'.join(code_lines)
            story.append(Paragraph(f'<pre>{code_text}</pre>', style_code))
            code_lines = []

    def flush_table():
        nonlocal table_rows, table_header, in_table
        if table_rows:
            story.append(Spacer(1, 2*mm))
            story.append(make_table(table_header, table_rows))
            story.append(Spacer(1, 3*mm))
            table_rows = []
            table_header = []
            in_table = False

    def flush_quote():
        nonlocal quote_lines, in_quote
        if quote_lines:
            qt = '<br/>'.join(quote_lines)
            story.append(Paragraph(qt, style_quote))
            quote_lines = []
            in_quote = False

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.startswith('```'):
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Table
        if line.startswith('|') and line.strip().endswith('|'):
            flush_quote()
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if not in_table:
                # Check next line for separator
                if i+1 < len(lines) and re.match(r'^\|[\s\-:|]+\|$', lines[i+1]):
                    table_header = cells
                    in_table = True
                    i += 2
                    continue
                else:
                    # Not a table, render as code
                    story.append(Paragraph(line.replace(' ', '&nbsp;'), style_code))
                    i += 1
                    continue
            else:
                # Check if still table
                if i+1 < len(lines) and not lines[i+1].startswith('|'):
                    table_rows.append(cells)
                    flush_table()
                else:
                    table_rows.append(cells)
                i += 1
                continue
        elif in_table:
            flush_table()

        # Quote
        if line.startswith('> '):
            flush_code()
            if not in_quote:
                in_quote = True
            quote_lines.append(line[2:])
            i += 1
            continue
        elif in_quote:
            flush_quote()

        # Headers
        if line.startswith('# ') and not line.startswith('## '):
            flush_code(); flush_quote()
            story.append(Paragraph(line[2:], style_title))
            story.append(Spacer(1, 2*mm))
        elif line.startswith('## ') and not line.startswith('### '):
            flush_code(); flush_quote()
            story.append(Paragraph(line[3:], style_h1))
        elif line.startswith('### '):
            flush_code(); flush_quote()
            story.append(Paragraph(line[4:], style_h2))
        elif line.startswith('#### '):
            flush_code(); flush_quote()
            story.append(Paragraph(line[5:], style_h3))

        # Horizontal rule
        elif line.strip() == '---':
            flush_code(); flush_quote()
            story.append(hr())

        # Bullet list
        elif line.startswith('- ') or line.startswith('  - '):
            flush_code(); flush_quote()
            indent_level = len(line) - len(line.lstrip())
            text_content = line.lstrip('- ')

            # Bold handling
            text_content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text_content)
            text_content = re.sub(r'`(.+?)`', r'<font face="DejaVuMono" size="8">\1</font>', text_content)

            s = ParagraphStyle('BulletTmp', parent=style_bullet,
                               leftIndent=12 + indent_level * 3,
                               bulletIndent=4 + indent_level * 3)
            story.append(Paragraph(f'• {text_content}', s))

        # Numbered list
        elif re.match(r'^\d+\.\s', line):
            flush_code(); flush_quote()
            text_content = re.sub(r'^\d+\.\s', '', line)
            text_content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text_content)
            text_content = re.sub(r'`(.+?)`', r'<font face="DejaVuMono" size="8">\1</font>', text_content)
            story.append(Paragraph(f'• {text_content}', style_bullet))

        # Empty line
        elif not line.strip():
            flush_code(); flush_quote()
            story.append(Spacer(1, 1*mm))

        # Regular paragraph
        else:
            flush_code(); flush_quote()
            text_content = line
            # Bold
            text_content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text_content)
            # Inline code
            text_content = re.sub(r'`([^`]+)`', r'<font face="DejaVuMono" size="8">\1</font>', text_content)
            # Arrows
            text_content = text_content.replace('→', '→ ')
            text_content = text_content.replace('►', '► ')

            story.append(Paragraph(text_content, style_body))

        i += 1

    # Flush remaining
    flush_code()
    flush_table()
    flush_quote()

    # ── 构建 PDF ──────────────────────────────────────────
    def on_first_page(canvas, doc):
        canvas.saveState()
        canvas.setFont('CJK', 8)
        canvas.setFillColor(grey)
        canvas.drawString(20*mm, 12*mm, 'VRAG — Vision-Augmented RAG Agent')
        canvas.drawRightString(A4[0] - 20*mm, 12*mm, '2026-05-03')
        canvas.restoreState()

    def on_later_pages(canvas, doc):
        canvas.saveState()
        canvas.setFont('CJK', 8)
        canvas.setFillColor(grey)
        canvas.drawString(20*mm, 12*mm, 'VRAG 创新方案文档')
        canvas.drawRightString(A4[0] - 20*mm, 12*mm, f'第 {doc.page} 页')
        # Header line
        canvas.setStrokeColor(BORDER_COLOR)
        canvas.setLineWidth(0.5)
        canvas.line(20*mm, A4[1] - 15*mm, A4[0] - 20*mm, A4[1] - 15*mm)
        canvas.restoreState()

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f'✅ PDF 已生成: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)')


if __name__ == '__main__':
    md_path = os.path.join(os.path.dirname(__file__), 'VRAG_创新方案文档.md')
    pdf_path = os.path.join(os.path.dirname(__file__), 'VRAG_创新方案文档.pdf')
    build_pdf(md_path, pdf_path)
