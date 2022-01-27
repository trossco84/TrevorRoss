from cgi import test
import pandas as pd
import numpy as np
from urllib.request import urlopen

test_url = "https://drive.google.com/open?id=1zDm0rAEdmbzSfwxQxCpWv8AtBCytAzAo"

page = urlopen(test_url)
html = page.read().decode("utf-8")
page.data()

PdfFileReader(file("http://example.com/a.pdf", "rb"))


import io
from urllib.request import Request, urlopen

from PyPDF2 import PdfFileReader


def get_pdf_from_url(url):
    """
    :param url: url to get pdf file
    :return: PdfFileReader object
    """
    remote_file = urlopen(Request(url)).read()
    memory_file = io.BytesIO(remote_file)
    pdf_file = PdfFileReader(memory_file)
    return pdf_file

get_pdf_from_url(test_url)

import urllib

url = test_url
webFile = urlopen(url)

pdfFile = open(url.split('/')[-1], 'w')
pdfFile.write(webFile.read().decode("utf-8"))
webFile.close()
pdfFile.close()

base = os.path.splitext(pdfFile)[0]
os.rename(pdfFile, base + ".pdf")

input1 = PdfFileReader(file(pdfFile, "rb"))