# Encode the corpus into embeddings
import faiss
import openai
from sentence_transformers import SentenceTransformer

from config import AZURE_OPENAI_DEPLOYMENT_NAME

# Define the RAG pipeline
class RAGPipeline:
    def __init__(self, web_page_content):
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight Sentence-BERT model
        self.web_page_content = web_page_content
        self.embeddings = self.convert_web_content_to_embeddings()
        self.index = self.index_embeddings()

    # Encode the corpus into embeddings
    def convert_web_content_to_embeddings(self):
        embeddings = self.retriever.encode(self.web_page_content, convert_to_tensor=False)
        return embeddings

    # Index the embeddings using FAISS
    def index_embeddings(self):
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)
        return index

    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k most relevant documents from the web_page_content using FAISS.
        """
        query_embedding = self.retriever.encode(query, convert_to_tensor=False)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        retrieved_docs = [self.web_page_content[idx] for idx in indices[0]]

        # Deduplicate the documents
        unique_docs = []
        seen = set()
        for doc in retrieved_docs:
            if doc not in seen:
                unique_docs.append(doc)
                seen.add(doc)

        return unique_docs[:top_k]

    @staticmethod
    def generate(query, retrieved_docs):
        """
        Generate an answer using Azure OpenAI, given the query and retrieved documents.
        """
        context = "\n".join(retrieved_docs)  # Combine retrieved documents as context
        prompt = f"""
        You are a expert in flight assistance. Use the following context to answer the question:

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4200,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()

    def __call__(self, query, top_k=3):
        """
        End-to-end RAG pipeline: retrieve information and generate an answer.
        """
        retrieved_docs = self.retrieve(query, top_k=top_k)
        if not retrieved_docs:
            return {"query": query, "retrieved_docs": [], "answer": "No relevant information found."}

        answer = self.generate(query, retrieved_docs)
        return {"query": query, "retrieved_docs": retrieved_docs, "answer": answer}