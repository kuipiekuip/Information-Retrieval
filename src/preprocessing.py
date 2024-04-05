from bs4 import BeautifulSoup
import requests
import re
from typing import List


def download_all_documents(url_list : List[str]) -> List[str]:
    return list(map(download_document, url_list))

def download_document(url : str) -> requests.models.Response:
    try:
        response = requests.get(url, timeout=120)
    except:
        print("Exception: Connection Problem; skipping file")
        return ""

    return response.content

def clean_all_documents(page_list : List[requests.models.Response]):
    return list(map(clean_document, page_list))

def clean_document(content : str):
    soup = BeautifulSoup(content, "html.parser")
    
    body_text = ""
    for tag in soup.find_all(['h1','p']):
        body_text += " " + tag.text
    return ' '.join(remove_punctuation(body_text).lower().split())

def remove_punctuation(text : str) -> str:
    return str(re.sub(r'[^\s\w](?![\w])|(?<![\w])[^\s\w]|^[^\s\w]|[^\s\w]$|\n|\xa0|\[|\]', '', text))