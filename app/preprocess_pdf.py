"""
PRE-PROCESSING PDFs FOR RAG
---------------------------
Requiere: pip install pdfplumber fpdf regex unidecode
"""

import re, os, pdfplumber
from unidecode import unidecode
from fpdf import FPDF
from pathlib import Path

from fpdf import FPDF
FONT_PATH = "./fonts/DejaVuSans.ttf"   # asegúrate de que exista

# ==== ajustes =====
RAW_DIR   = Path("raw_pdfs")       # carpeta con los pdf originales
CLEAN_DIR = Path("clean_pdfs")     # salidas
EXCLUDE_PATTERNS = [
    r"^Página \d+ de \d+$",
    r"^CONSEJO (SUPERIOR|ACADÉMICO)",  # encabezados repetidos
    r"^Facultad", r"^INVITADOS$",
]
YEAR_TAG_RE = re.compile(r"20[2-4]\d")  # 2020-2024
# ===================

def detect_year(filename, first_pages_text):
    # 1) intente filename: Informe-2021..., 2) busque en texto
    m = re.search(r"20\d{2}", filename)
    if m: return m.group()
    m = YEAR_TAG_RE.search(first_pages_text)
    return m.group() if m else "XXXX"

def is_noise(line):
    line_clean = line.strip()
    if not line_clean or len(line_clean) < 4:
        return True
    return any(re.match(p, line_clean, re.I) for p in EXCLUDE_PATTERNS)

def clean_pdf(path: Path):
    with pdfplumber.open(path) as pdf:
        cleaned_lines = []
        first_pages_text = ""
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if page_num < 2:
                first_pages_text += text + "\n"
            for ln in text.splitlines():
                if not is_noise(ln):
                    cleaned_lines.append(re.sub(r"\s+", " ", ln).strip())
        year = detect_year(path.name, first_pages_text)
        header = f"### VIGENCIA {year} – {'PLAN DE DESARROLLO' if 'PDI' in path.name.upper() else 'INFORME DE GESTIÓN'}\n\n"
        return header + "\n".join(cleaned_lines)

# def save_as_pdf(text:str, out_path:Path):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Times", size=11)
#     pdf.add_page()
#     for line in text.split("\n"):
#         pdf.multi_cell(0, 5, txt=line)
#     pdf.output(out_path)

def save_as_pdf(text:str, out_path:Path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
    pdf.set_font("DejaVu", size=11)
    pdf.add_page()
    for line in text.split("\n"):
        pdf.multi_cell(0, 5, txt=line)
    pdf.output(out_path)


def main():
    for pdf_file in RAW_DIR.glob("*.pdf"):
        clean_text = clean_pdf(pdf_file)
        rel = pdf_file.stem + "_clean.pdf"
        out_pdf = CLEAN_DIR / rel
        out_txt = CLEAN_DIR / (pdf_file.stem + ".txt")
        out_pdf.parent.mkdir(parents=True, exist_ok=True)

        # save_as_pdf(clean_text, out_pdf)
        out_txt.write_text(clean_text, encoding="utf-8")
        print(f"✔ {pdf_file.name} -> {out_pdf.name}")

if __name__ == "__main__":
    main()
