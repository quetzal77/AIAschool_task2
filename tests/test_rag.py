from rag_src.rag_tool import RAGPipeline
from rag_src.webpage_scrapper import scrape_web_pages

# Web pages using as datasource
web_pages = [
    # "https://help.ryanair.com/hc/en-gb/categories/12489112419089-Bag-Rules",
    "https://www.nidirect.gov.uk/articles/air-travel-hand-baggage-and-hold-luggage-rules"
]

# Web page content retrieval
web_page_content = scrape_web_pages(web_pages)

# User query
query = "What is a maximum weight of hand language?"

# Instantiate RAG pipeline
rag_pipeline = RAGPipeline(web_page_content = web_page_content)

# Generate LLM answer
answer = rag_pipeline(query, top_k=3)

# Print the results
print("Query:", answer['query'])

print("Retrieved Documents:")
for i, doc in enumerate(answer['retrieved_docs']):
     print(f"[{i+1}] {doc[:300]}...")  # Show the first 300 characters of each document

print("\nGenerated Answer:", answer['answer'])