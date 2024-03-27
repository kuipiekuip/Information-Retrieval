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
    for tag in soup.find_all(['h1','p']):
        body_text += " " + tag.text
    return ' '.join(remove_punctuation(body_text).lower().split())

def remove_punctuation(text : str) -> str:
    return str(re.sub(r'[^\s\w](?![\w])|(?<![\w])[^\s\w]|^[^\s\w]|[^\s\w]$|\n|\xa0|\[|\]', '', text))

url_list = ["https://nos.nl/artikel/2513343-unilever-schrapt-7500-banen-en-zet-ijsjes-in-de-etalage", 
            "https://nos.nl/artikel/2513359-trump-krijgt-borgsom-van-454-miljoen-dollar-niet-rond-onmogelijke-opdracht",
            "https://www.theguardian.com/environment/2024/mar/19/air-pollution-health-report",
            "https://en.wikipedia.org/wiki/Information_retrieval",
            "https://www.geeksforgeeks.org/what-is-information-retrieval/",
            "https://www.wis.ewi.tudelft.nl/information-retrieval",
            "https://www.coveo.com/blog/information-retrieval/",
            "https://www.elastic.co/what-is/information-retrieval",
            "https://studiegids.universiteitleiden.nl/courses/105168/information-retrieval"]

page_list = download_all_documents(url_list)
text_list = clean_all_documents(page_list)
#print(text_list)