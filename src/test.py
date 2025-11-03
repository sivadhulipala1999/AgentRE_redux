# import requests
# from googlesearch import search
# from bs4 import BeautifulSoup


# def extract_info_from_url(url):
#     try:
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(url, headers=headers, timeout=5)
#         soup = BeautifulSoup(response.text, 'html.parser')

#         paragraphs = soup.find_all('p')
#         text = ' '.join(p.get_text()
#                         for p in paragraphs[:5])  # first few paragraphs
#         return text
#     except Exception as e:
#         return f"Error extracting from {url}: {e}"


# query = "OpenAI GPT-4 site:en.wikipedia.org"
# num_results = 5
# results = list(search(query, num_results=num_results))

# for url in results:
#     print(f"--- Extracting from: {url} ---")
#     info = extract_info_from_url(url)
#     print(info[:500], "\n")  # preview


import wikipediaapi

wiki = wikipediaapi.Wikipedia(user_agent="AgentRE_Agent", language="en")
page = wiki.page("ABC Entertainment")

# Get the summary/lead
print(page.summary)
