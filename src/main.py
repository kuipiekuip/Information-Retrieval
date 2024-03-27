from geturl import GetURLs
from preprocessing import download_all_documents, clean_all_documents
from vectorizer import run_query_expansion

def main():
    list = GetURLs("software", 5, True)
    page_list = download_all_documents(list)
    text_list = clean_all_documents(page_list)

    print(text_list)
    print(len(text_list))


if __name__== "main_" :
    main()