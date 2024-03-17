import requests

subscription_key = "c3c922fcb1394850b975f21bc84fdb64"

endpoint1 = "https://api.bing.microsoft.com/v7.0/search"
endpoint2 = "https://api.bing.microsoft.com/v7.0/news/search" #this doesn't work, don't know why


def GetURLs(query : str, N : int, flag : bool):

    # Construct a request
    mkt = 'en-US'
    url = endpoint1
    if flag:
        url = endpoint2

    headers = { "Ocp-Apim-Subscription-Key" : subscription_key }
    params = { 'q': query, 'mkt': mkt, 'count': N}

    # Call the API
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        d = response.json()
        r = d['webPages']['value']
        return [item['url'] for item in r]

    except Exception as ex:
        raise ex