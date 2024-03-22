import requests
import json
import os

subscription_key = "4e4ba1ec06b0413c92afe897d6fe4d3a"

endpoint1 = "https://api.bing.microsoft.com/v7.0/search" # normal search
endpoint2 = "https://api.bing.microsoft.com/v7.0/news/search" # news search

# get the top N urls given the query
# flag indicates the option of "normal search"(False) or "news search"(True)
def GetURLs(query : str, N : int, flag : bool):
    # Read from the file
    if flag:
        filename = "log2.txt"
    else:
        filename = "log1.txt"

    if os.path.exists(filename) and os.stat(filename).st_size != 0:
        with open(filename) as f:
            log = json.load(f)
        list = log.get(query, [])
        if len(list) >= N:
            return list[:N]
    else:
        log = {}

    # Construct a request
    mkt = 'en-US'
    if flag:
        url = endpoint2
    else:
        url = endpoint1

    headers = { "Ocp-Apim-Subscription-Key" : subscription_key }
    params = { 'q': query, 'mkt': mkt, 'count': N}

    # Call the API
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        d = response.json()

        if flag:
            r = d['value']
        else:
            r = d['webPages']['value']

        log[query] = [item['url'] for item in r]

        with open(filename, 'w') as f: 
            f.write(json.dumps(log))

        return log[query]

    except Exception as ex:
        raise ex