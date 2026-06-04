import os
import fitz  # PyMuPDF

def convert_pdf_to_transparent_png(pdf_filename, output_filename="output.png"):
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths for the same directory operation
    pdf_path = os.path.join(current_dir, pdf_filename)
    output_path = os.path.join(current_dir, output_filename)
    
    # Verify the input PDF exists in this directory
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find '{pdf_filename}' in {current_dir}")
    
    # Open the PDF document
    doc = fitz.open(pdf_path)
    page = doc[0]  # Standalone documents are single-page (index 0)
    
    # Target ~300 DPI for presentation sharpness (300 / 72 = 4.1666...)
    zoom_factor = 300 / 72
    matrix = fitz.Matrix(zoom_factor, zoom_factor)
    
    # Render the page to a pixel map with an alpha channel for transparency
    pix = page.get_pixmap(matrix=matrix, alpha=True)
    
    # Save the file to disk
    pix.save(output_path)
    print(f"Success! High-resolution transparent image saved to: {output_path}")

if __name__ == "__main__":
    # Change this to match your actual compiled LaTeX PDF filename
    target_pdf = r"C:\Users\hp\Downloads\equationmaker (5).pdf"
    
    try:
        convert_pdf_to_transparent_png(target_pdf, "latex_presentation_asset.png")
    except Exception as e:
        print(f"Error: {e}")