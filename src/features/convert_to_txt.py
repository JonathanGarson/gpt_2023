"""
This code converts all .docx files in a directory to .txt files. It uses the pydocx library which read the xml content beneath a docx file and save the file as a .txt file.
The procedure is faster than in 'convert_to_txt', count 15s for 206 files (0.07s per file).
"""

import os 
import glob
from tqdm import tqdm
from docx import Document

def correct_docx_file_path(file_path):
    """
    Correct the file path if it ends with .docx.docx instead of .docx

    Args:
        file_path (str): The path to the file.
    """
    filename = os.path.basename(file_path)
    if filename.endswith(".docx.docx"):
        os.rename(file_path, file_path.replace(".docx.docx", ".docx"))
    else:
        pass


def extract_text_from_docx(file_path, output_directory):
    """
    Extract text from a docx file and save it in a txt file.

    Args:
        file_path (str): The path to the docx file.
        output_directory (str): The path to the directory where the txt file will be saved.
    """
    doc = Document(file_path)
    text = []

    # Extraction des textes des paragraphes
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Ecriture du texte extrait dans un nouveau fichier txt
    with open(os.path.join(output_directory, os.path.basename(file_path)[:-5] + ".txt"), "w", encoding="utf-8") as f:
        for line in text:
            f.write(line + "\n")

# Set your paths
directory = r"data/text/docx"
docx_files = glob.glob(directory + "/*.docx")
output_directory = r"data/text/txt"

if not docx_files:
    print("No .docx files found in the specified directory.")
else:
    for file_path in tqdm(docx_files, desc="Extracting text from docx files", unit="files"):
        print("le dossier:", file_path, "a été converti")
        correct_docx_file_path(file_path)
        extract_text_from_docx(file_path, output_directory)

print("=========")
print("All done!")