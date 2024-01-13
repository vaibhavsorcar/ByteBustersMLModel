# Code to accept a pdf file , read the contents and produce back a pdf file.
# This method is used language translation , so as to read the content from english and translate to the language of our need.

import fitz  # PyMuPDF
from fpdf import FPDF

# Open a PDF file using PyMuPDF
file_path = "./images_dataset/R&A.pdf"
pdf_document_handle = fitz.open(file_path)

# Get the number of pages in the PDF
num_pages = pdf_document_handle.page_count

# Extract text from each page
pdf_text_content = ""
for page_num in range(num_pages):
    page = pdf_document_handle.load_page(page_num)
    text = page.get_text()
    pdf_text_content += text

# Save extracted text to a text file
text_output_path = "12.txt"
with open(text_output_path, "w", encoding="utf-8") as text_file:
    text_file.write(pdf_text_content)

# Close the PDF document handle
pdf_document_handle.close()

# Create a PDF using FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)  # Change font to Arial

# Read text from the saved file and add it to the PDF
with open(text_output_path, "r", encoding="utf-8") as text_file:
    for line in text_file:
        # Handle Unicode characters
        line_encoded = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=line_encoded)

#Save the output PDF
pdf_output_path = "123.pdf"
pdf.output(pdf_output_path)
