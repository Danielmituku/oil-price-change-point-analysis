"""
Convert Markdown report to PDF using WeasyPrint.

Usage:
    python scripts/md_to_pdf.py [input_md] [output_pdf]
    
    Default: converts reports/interim_report.md to reports/interim_report.pdf
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import sys
import os

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

# CSS styling for the PDF
CSS_STYLE = """
@page {
    size: A4;
    margin: 2cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 10px;
        color: #666;
    }
}

body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
}

h1 {
    color: #1a5276;
    font-size: 24pt;
    border-bottom: 3px solid #1a5276;
    padding-bottom: 10px;
    margin-top: 0;
}

h2 {
    color: #2874a6;
    font-size: 16pt;
    border-bottom: 1px solid #2874a6;
    padding-bottom: 5px;
    margin-top: 25px;
}

h3 {
    color: #2e86ab;
    font-size: 13pt;
    margin-top: 20px;
}

h4 {
    color: #444;
    font-size: 11pt;
    margin-top: 15px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 10pt;
}

th {
    background-color: #2874a6;
    color: white;
    padding: 10px 8px;
    text-align: left;
    font-weight: bold;
}

td {
    padding: 8px;
    border-bottom: 1px solid #ddd;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

tr:hover {
    background-color: #e8f4f8;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 9pt;
}

pre {
    background-color: #f4f4f4;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9pt;
    border-left: 4px solid #2874a6;
}

pre code {
    background-color: transparent;
    padding: 0;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
    border: 1px solid #ddd;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

blockquote {
    border-left: 4px solid #2874a6;
    margin: 15px 0;
    padding: 10px 20px;
    background-color: #f8f9fa;
    font-style: italic;
}

hr {
    border: none;
    border-top: 2px solid #ddd;
    margin: 30px 0;
}

ul, ol {
    margin: 10px 0;
    padding-left: 30px;
}

li {
    margin: 5px 0;
}

strong {
    color: #1a5276;
}

/* Executive Summary styling */
p:first-of-type {
    font-size: 11pt;
}

/* Table of contents styling */
a {
    color: #2874a6;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Page break before major sections */
h2 {
    page-break-before: auto;
}

/* Avoid orphans and widows */
p {
    orphans: 3;
    widows: 3;
}

/* Keep headings with following content */
h1, h2, h3, h4 {
    page-break-after: avoid;
}

/* Figure captions */
.figure-caption {
    text-align: center;
    font-style: italic;
    font-size: 10pt;
    color: #666;
    margin-top: -10px;
}
"""


def convert_md_to_pdf(input_md: Path, output_pdf: Path):
    """
    Convert a Markdown file to PDF.
    
    Parameters
    ----------
    input_md : Path
        Path to the input Markdown file
    output_pdf : Path
        Path for the output PDF file
    """
    print(f"Converting: {input_md}")
    print(f"Output: {output_pdf}")
    
    # Read the markdown content
    with open(input_md, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    # Enable extensions for tables, fenced code, etc.
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'nl2br',
            'sane_lists'
        ]
    )
    
    # Get the directory of the markdown file for relative image paths
    md_dir = input_md.parent
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Interim Report - Brent Oil Price Analysis</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert to PDF using WeasyPrint
    # Set base_url to the markdown file's directory for relative image paths
    html = HTML(string=full_html, base_url=str(md_dir))
    css = CSS(string=CSS_STYLE)
    
    print("Generating PDF...")
    html.write_pdf(output_pdf, stylesheets=[css])
    
    print(f"\n✅ PDF generated successfully!")
    print(f"   Location: {output_pdf}")
    print(f"   Size: {output_pdf.stat().st_size / 1024:.1f} KB")


def main():
    """Main function to run the conversion."""
    # Default paths
    default_input = REPORTS_DIR / "interim_report.md"
    default_output = REPORTS_DIR / "interim_report.pdf"
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        input_md = Path(sys.argv[1])
    else:
        input_md = default_input
    
    if len(sys.argv) >= 3:
        output_pdf = Path(sys.argv[2])
    else:
        output_pdf = default_output
    
    # Validate input file exists
    if not input_md.exists():
        print(f"❌ Error: Input file not found: {input_md}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    convert_md_to_pdf(input_md, output_pdf)


if __name__ == "__main__":
    main()
