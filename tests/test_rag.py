from rag_src.rag_tool import RAGPipeline
from rag_src.webpage_scrapper import scrape_web_pages

# Web pages using as datasource
web_pages = [
    # "https://help.ryanair.com/hc/en-gb/categories/12489112419089-Bag-Rules",
    "https://www.booking.com/guides/article/flights/checked-baggage-rules-for-flying.html?aid=318615;label=New_English_EN_HR_27034598665-O5*YTBci*AKCr1pOlzFaMAS634117826559:pl:ta:p1:p2:ac:ap:neg:fi:tidsa-55482331735:lp1028863:li:dec:dm:ag27034598665:cmp400849465;ws=&gad_source=1&gad_campaignid=400849465&gbraid=0AAAAAD_Ls1Iak1oFoh1dmbPkVrJxS_xvO&gclid=Cj0KCQjw-NfDBhDyARIsAD-ILeBFZaUUdwJxzhKU_PKde8ScClYfzRCV1UOuRkgs3xbvcgY4d7p3-bwaAswbEALw_wcB"
]

# Web page content retrieval
web_page_content = scrape_web_pages(web_pages)

# User query
query = "What is a maximum dimensions of language in centimeters?"

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