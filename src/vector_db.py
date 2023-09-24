import os
import openai
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import pprint

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import WebBaseLoader


import web_extractor



# Specify the path to config.json (adjust the path as needed)
config_file_path = '/Users/jasonz/forward_data_lab_llmie/forward_data-llm_ie/config.json'

with open(config_file_path, "r") as config_file:
    config = json.load(config_file)
    openai.api_key = config["api_key"]

llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)



def retrieve_information_from_url(url: str, query: str) -> str:
    
    # Load the document, split it into chunks, embed each chunk, and load it into the vector store
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)
    db = Chroma.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai.api_key))
    
    # Search for similarity based on the query
    docs_sim = db.similarity_search(query)
    
    # Retrieve and return the page content
    if docs_sim:
        return docs_sim[0].page_content
    else:
        return "Information not found."

# Example usage:
url = "https://blog.logrocket.com/implement-webassembly-webgl-viewer-using-rust/"
query = "How to Setup environment?"
api_key = "your_openai_api_key_here"

answer = retrieve_information_from_url(url, query)
print(answer)