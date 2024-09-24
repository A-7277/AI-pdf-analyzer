import streamlit as st
import google.generativeai  as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()
# st.text(os.getenv('api_key'))
# os.environ['GOOGLE_API_KEY']=os.getenv('api_key')

with st.sidebar:
  st.subheader('PDF analyzer and Q/A chatbot') 
  file=st.file_uploader('upload your PDF here!',type='pdf')

if file:
  with open('temp.pdf', 'wb') as f:
    f.write(file.read())
  loader=PyPDFLoader('temp.pdf')
  data=loader.load()
  splitter=RecursiveCharacterTextSplitter(separators=['\n','.',' '],chunk_size=1000,chunk_overlap=200)
  chunks=splitter.split_documents(documents=data)
  
  model_name = "sentence-transformers/all-mpnet-base-v2"  
  model_kwargs = {'device': 'cpu'}
  encode_kwargs = {'normalize_embeddings': False}
  hf = HuggingFaceEmbeddings(
      model_name=model_name,
      model_kwargs=model_kwargs,
      encode_kwargs=encode_kwargs
  )
  
  vectorindex=FAISS.from_documents(chunks,hf)
  retriever=vectorindex.as_retriever()
  
  messages = st.container(border=True)
  if prompt := st.chat_input("Say something"):
      messages.chat_message("user").write(prompt)
      para=retriever.get_relevant_documents(prompt)
      
      genai.configure(api_key=os.getenv('api_key'))
       
      generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
      }

      model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=f"find answer of this question- {prompt} from this paragarph {para}",
      )
      
      messages.chat_message("assistant").write(f"Echo: {(model.generate_content(prompt)).text}")
   