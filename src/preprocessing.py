from bs4 import BeautifulSoup
import requests
import re
from typing import List


def download_all_documents(url_list : List[str]) -> List[requests.models.Response]:
    return list(map(requests.get, url_list))

def clean_all_documents(page_list : List[requests.models.Response]):
    return list(map(clean_document, page_list))

def clean_document(page : requests.models.Response):
    soup = BeautifulSoup(page.content, "html.parser")
    
    body_text = ""
    for tag in soup.find_all(['p']):
        body_text += " " + tag.text

    return remove_punctuation(body_text).lower()

def remove_punctuation(text : str) -> str:
    return re.sub(r'[^\w\s\W]|(?<!\w)\W|\W(?!\w)', '', text)

url_list = ["https://nos.nl/artikel/2513343-unilever-schrapt-7500-banen-en-zet-ijsjes-in-de-etalage", 
            "https://nos.nl/artikel/2513359-trump-krijgt-borgsom-van-454-miljoen-dollar-niet-rond-onmogelijke-opdracht",
            "https://www.theguardian.com/environment/2024/mar/19/air-pollution-health-report"]

page_list = download_all_documents(url_list)
text_list = clean_all_documents(page_list)
print(text_list)