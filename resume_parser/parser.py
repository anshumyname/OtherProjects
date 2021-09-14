from PIL import Image
import pytesseract
import sys
from pdf2image import convert_from_path
import os

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
filenames = os.listdir("./dataset/testResumes/")

for file in filenames:
    PDF_file = "./dataset/testResumes/"+file
    OUT_file = './test_txt/' +file[:-4]+ '.txt'
    pages = convert_from_path(PDF_file, 500)
    f = open(OUT_file, "a")
    text_wr = ""
    for idx, page in enumerate(pages):
        filename = "page_"+ str(idx) + ".jpg"
        page.save(filename, 'PNG')
        text = str(((pytesseract.image_to_string(Image.open(filename)))))
        # text = text.replace('-\n', '')
        text_wr+=text

    f.write(text_wr)
    f.close()
    

