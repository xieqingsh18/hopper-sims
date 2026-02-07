#!/usr/bin/env python3
"""
PDF to HTML/Text Converter

This script converts PDF files to HTML or plain text format,
making it easier to extract documentation from PDF files.
"""

import sys
import os
from pathlib import Path
from typing import Optional


def convert_pdf_to_text_pdfplumber(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert PDF to text using pdfplumber (most reliable).

    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save text file

    Returns:
        Extracted text content
    """
    try:
        import pdfplumber
    except ImportError:
        print("Installing pdfplumber...")
        os.system("pip install pdfplumber -q")
        import pdfplumber

    print(f"Converting {pdf_path} to text...")

    text_content = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"  Processing page {page_num + 1}/{len(pdf.pages)}...")
            text = page.extract_text()
            if text:
                text_content.append(f"=== PAGE {page_num + 1} ===\n\n{text}\n\n")

    result = "\n".join(text_content)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"  Saved to {output_path}")

    return result


def convert_pdf_to_text_pymupdf(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert PDF to text using PyMuPDF (fitz) - faster option.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save text file

    Returns:
        Extracted text content
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Installing PyMuPDF (fitz)...")
        os.system("pip install pymupdf -q")
        import fitz

    print(f"Converting {pdf_path} to text...")

    doc = fitz.open(pdf_path)
    text_content = []

    for page_num in range(len(doc)):
        print(f"  Processing page {page_num + 1}/{len(doc)}...")
        page = doc[page_num]
        text = page.get_text()
        text_content.append(f"=== PAGE {page_num + 1} ===\n\n{text}\n\n")

    doc.close()

    result = "\n".join(text_content)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"  Saved to {output_path}")

    return result


def convert_pdf_to_html(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert PDF to HTML using pdfplumber with layout preservation.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save HTML file

    Returns:
        HTML content
    """
    try:
        import pdfplumber
    except ImportError:
        print("Installing pdfplumber...")
        os.system("pip install pdfplumber -q")
        import pdfplumber

    print(f"Converting {pdf_path} to HTML...")

    html_parts = ['<!DOCTYPE html>\n<html>\n<head>\n']
    html_parts.append('<meta charset="UTF-8">\n')
    html_parts.append('<style>\n')
    html_parts.append('  body { font-family: Arial, sans-serif; margin: 40px; }\n')
    html_parts.append('  .page { margin-bottom: 40px; border: 1px solid #ccc; padding: 20px; }\n')
    html_parts.append('  .page-number { font-size: 18px; font-weight: bold; color: #333; }\n')
    html_parts.append('  pre { background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }\n')
    html_parts.append('</style>\n')
    html_parts.append('</head>\n<body>\n')

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"  Processing page {page_num + 1}/{len(pdf.pages)}...")
            html_parts.append(f'<div class="page">\n')
            html_parts.append(f'  <div class="page-number">Page {page_num + 1}</div>\n')

            text = page.extract_text()
            if text:
                # Convert to HTML-safe format
                text_html = text.replace('<', '&lt;').replace('>', '&gt;')
                text_html = text_html.replace('\n', '<br>\n')
                html_parts.append(f'  <pre>{text_html}</pre>\n')

            html_parts.append('</div>\n')

    html_parts.append('</body>\n</html>')

    result = '\n'.join(html_parts)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"  Saved to {output_path}")

    return result


def convert_pdf_to_markdown(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert PDF to Markdown format.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save markdown file

    Returns:
        Markdown content
    """
    try:
        import pdfplumber
    except ImportError:
        print("Installing pdfplumber...")
        os.system("pip install pdfplumber -q")
        import pdfplumber

    print(f"Converting {pdf_path} to markdown...")

    md_content = [f"# CUDA Runtime API Documentation\n\n"]
    md_content.append(f"Converted from: {pdf_path}\n\n")

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"  Processing page {page_num + 1}/{len(pdf.pages)}...")
            md_content.append(f"## Page {page_num + 1}\n\n")

            text = page.extract_text()
            if text:
                # Convert to markdown-friendly format
                lines = text.split('\n')
                for line in lines:
                    # Preserve indentation
                    line = line.replace('<', '&lt;').replace('>', '&gt;')
                    md_content.append(f"{line}\n")

                md_content.append("\n---\n\n")

    result = ''.join(md_content)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"  Saved to {output_path}")

    return result


def extract_code_signatures(pdf_path: str) -> None:
    """
    Extract CUDA function signatures from PDF and save to separate file.

    Looks for patterns like:
    - cudaError_t cudaMalloc(void** devPtr, size_t size)
    - cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
    """
    try:
        import pdfplumber
    except ImportError:
        print("Installing pdfplumber...")
        os.system("pip install pdfplumber -q")
        import pdfplumber

    print(f"Extracting CUDA API signatures from {pdf_path}...")

    api_signatures = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"  Scanning page {page_num + 1}/{len(pdf.pages)}...")
            text = page.extract_text()

            # Look for CUDA function signatures
            lines = text.split('\n')
            for line in lines:
                line = line.strip()

                # Look for common CUDA API patterns
                if 'cudaError_t cuda' in line and '(' in line:
                    api_signatures.append(line)
                elif 'cuda' in line and ('(void*' in line or '(size_t' in line or '(int' in line):
                    api_signatures.append(line)

    # Save signatures
    output_path = pdf_path.replace('.pdf', '_signatures.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# CUDA Runtime API Signatures\n\n")
        f.write(f"# Extracted from: {pdf_path}\n\n")
        for sig in api_signatures:
            f.write(f"{sig}\n")

    print(f"\n  Extracted {len(api_signatures)} API signatures")
    print(f"  Saved to {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert PDF to HTML/Text/Markdown')
    parser.add_argument('pdf_file', help='Path to PDF file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', choices=['text', 'html', 'markdown', 'signatures'],
                      default='text', help='Output format (default: text)')
    parser.add_argument('--pymupdf', action='store_true',
                      help='Use PyMuPDF (fitz) instead of pdfplumber for text conversion')

    args = parser.parse_args()

    if not os.path.exists(args.pdf_file):
        print(f"Error: File not found: {args.pdf_file}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.pdf_file)[0]
        ext = '.html' if args.format == 'html' else '.md' if args.format == 'markdown' else '.txt'
        if args.format == 'signatures':
            ext = '_signatures.txt'
        output_path = base + ext

    # Convert based on format
    if args.format == 'html':
        convert_pdf_to_html(args.pdf_file, output_path)
    elif args.format == 'markdown':
        convert_pdf_to_markdown(args.pdf_file, output_path)
    elif args.format == 'signatures':
        extract_code_signatures(args.pdf_file)
    elif args.pymupdf:
        convert_pdf_to_text_pymupdf(args.pdf_file, output_path)
    else:
        convert_pdf_to_text_pdfplumber(args.pdf_file, output_path)

    print(f"\n✓ Conversion complete!")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
