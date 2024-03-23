import os
from llama_index.packs.raptor import RaptorPack
os.environ["OPENAI_API_KEY"] = "PUT YOUR KEY HERE"

import nest_asyncio
nest_asyncio.apply()
from llama_index.core import SimpleDirectoryReader

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.packs.raptor import RaptorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness

def prompt(q, vector_store):
    retriever = RaptorRetriever(
        [],
        embed_model=OpenAIEmbedding(
            model="text-embedding-3-small"
        ),  # used for embedding clusters
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
        vector_store=vector_store,  # used for storage
        similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
        mode="collapsed",  # sets default mode
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    )
    return str(query_engine.query(q))

import streamlit as st
def main():
    st.title("Chat with your CV!")
    
    uploaded_file = st.file_uploader("Upload your CV:")
    documents=""

    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        st.success("File has been uploaded and saved to 'uploads' folder.") 

    # Input text
    input_text = st.text_area("Enter your query:")
    ground_truth = st.text_area("Enter your expected answer:")
    processed_output=""
    
    client = chromadb.PersistentClient(path=".\cv_db")
    collection = client.get_or_create_collection("raptor")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    raptor_pack = RaptorPack(
            documents,
            embed_model=OpenAIEmbedding(
                model="text-embedding-3-small"
            ),  # used for embedding clusters
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
            vector_store=vector_store,  # used for storage
            similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
            mode="collapsed",  # sets default mode
            transformations=[
                SentenceSplitter(chunk_size=400, chunk_overlap=50)
            ],  # transformations applied for ingestion
        )
    # Button to process text
    if st.button("Submit"):
        if input_text:
            processed_output = prompt(input_text,vector_store)
            st.write("Processed Output:")
            st.success(processed_output)
        else:
            st.warning("Please enter some text before submitting.")
    
    # Button to validate output
    if st.button("Validate"):

        context = raptor_pack.run(input_text, mode="collapsed")[0].text
        
        if input_text:
            data_samples = {
                'question': [input_text],
                'answer': [processed_output],
                'contexts' : [[context]],
                'ground_truth': [ground_truth]
            }
            dataset = Dataset.from_dict(data_samples)
            score = evaluate(dataset,metrics=[answer_correctness])
            st.write("Answer Correctness:")
            st.write(score['answer_correctness'])
        else:
            st.warning("Please enter some text before validating.")

if __name__ == "__main__":
    main()