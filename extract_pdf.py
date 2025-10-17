#!/usr/bin/env python3
import PyPDF2
import sys

def extract_pdf_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            print(f"Number of pages: {len(pdf_reader.pages)}")
            print("=" * 50)
            
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                print(f"\n--- PAGE {page_num} ---")
                print(text)
                print("-" * 30)
                full_text += f"\n[PAGE {page_num}]\n{text}\n"
            
            return full_text
            
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

if __name__ == "__main__":
    pdf_path = "blood_pressure_guide.pdf"
    extract_pdf_text(pdf_path)