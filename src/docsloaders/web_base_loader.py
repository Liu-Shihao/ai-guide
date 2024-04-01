from langchain_community.document_loaders import WebBaseLoader
'''
pip install beautifulsoup4
https://python.langchain.com/docs/integrations/document_loaders/web_base
'''

loader = WebBaseLoader("https://gorilla.cs.berkeley.edu/blogs/9_raft.html")

data = loader.load()

print(data)

# Using proxies

# loader = WebBaseLoader(
#     "https://www.walmart.com/search?q=parrots",
#     proxies={
#         "http": "http://{username}:{password}:@proxy.service.com:6666/",
#         "https": "https://{username}:{password}:@proxy.service.com:6666/",
#     },
# )
# docs = loader.load()