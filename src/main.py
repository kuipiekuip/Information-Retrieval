from geturl import GetURLs
from preprocessing import download_all_documents, clean_all_documents
from vectorizer import run_query_expansion

def main():
    original_query = {"software"}
    list = GetURLs("software", 5, True)
    page_list = download_all_documents(list)
    text_list = clean_all_documents(page_list)
    output = run_query_expansion(text_list, original_query)

    print("==================================================================================")
    print(output)


if __name__== "main_" :
    main()