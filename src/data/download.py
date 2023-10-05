from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time
import pandas as pd
import os
from tqdm import tqdm

# File path
download_file = r"./data/raw/full_data_link_legifrance.xlsx"

# Input directory
list_url = pd.read_excel(download_file)["UrlLegifrance"].tolist()

# Code
prefs = {
    "download.default_directory": os.getcwd() + r"\data\text\docx", 
}

print(os.getcwd() + r"\data\text\docx")

for url in tqdm(list_url, desc="Downloading", unit="file"):
    try:
        options = webdriver.ChromeOptions()
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        html = driver.page_source
        soup = bs(html, 'html.parser')
        download_url = 'https://www.legifrance.gouv.fr' + soup.find_all(class_="doc-download")[0].attrs['href']
        driver.get(download_url)
        time.sleep(1)
        driver.quit()
    except Exception as e:
        print("Error occurred:", str(e))
        continue

print("====================================")
print("Download finished")
print("====================================")