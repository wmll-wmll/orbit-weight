#!/usr/bin/env python3
"""Export DOCX deliverables to PDF with Word, then render every page to PNG for visual QA."""
import os, sys, glob
import win32com.client as win32
import pypdfium2 as pdfium

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "deliverable")
QA = os.path.join(ROOT, "_qa_png")
os.makedirs(QA, exist_ok=True)


def to_pdf(docx_path):
    pdf = os.path.splitext(docx_path)[0] + ".pdf"
    app = win32.Dispatch("Word.Application")
    app.Visible = False
    doc = app.Documents.Open(docx_path, ReadOnly=True)
    doc.SaveAs(pdf, FileFormat=17)
    pages = doc.ComputeStatistics(2)
    doc.Close(False)
    app.Quit()
    return pdf, pages


def render(pdf, tag):
    doc = pdfium.PdfDocument(pdf)
    outs = []
    for i in range(len(doc)):
        page = doc[i]
        img = page.render(scale=1.6).to_pil()
        p = os.path.join(QA, f"{tag}_p{i+1:02d}.png")
        img.save(p)
        outs.append(p)
    return outs


if __name__ == "__main__":
    for name in sys.argv[1:] or ["main_clean", "supplementary",
                                 "Response_to_Reviewers", "cover_letter"]:
        dx = os.path.join(OUT, name + ".docx")
        pdf, pages = to_pdf(dx)
        imgs = render(pdf, name)
        print(f"{name}: {pages} pages -> {len(imgs)} PNG")
