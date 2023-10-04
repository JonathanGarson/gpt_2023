import os
import glob
from tqdm import tqdm

def clean_txt(file):
    """
    Clean the files of their \n, \t, \r, and make them lowercase.
    """
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        text = text.lower()

    # Save the cleaned text back to the file.
    with open(file, 'w', encoding='utf-8') as f:
        f.write(text)

# TO AUTOMATE THE CLEANING OF THE TXT FILES
# Path to the directory containing the .txt files
input_directory = r"./data/text/txt/"
output_directory = r"./data/text/txt/"  # Add a backslash at the end

# Use glob to find .txt files in the input directory
files = glob.glob(input_directory + '*.txt')

for file in tqdm(files, desc='Cleaning files', unit='files'):
    clean_txt(file)

