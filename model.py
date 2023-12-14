import requests
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import chainlit as cl
import asyncio

#console command: chainlit run model2.py -w
# Configuration Constants
DB_FAISS_PATH = 'vectorstore/db_faiss'
TOGETHER_AI_URL = "https://api.together.xyz/inference"
TOGETHER_AI_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
API_KEY = "9116a14c0461e8bb61afa5acc310a4d6f8c4a2358d1fe8c1040945106e8b5acb"

# Custom Prompt Template
custom_prompt_template = """
You are a chatbot that has access to context retrieved from a database of medical labels from the fda database of drugs. 
You are only allowed to answer question relating to various drugs and medications, and only with responses that can be referenced back to the context provided to you.
Make sure your responses are well structured, complete, answer the question with details, and end on a period. Be kind, when people say hello, greet them, and try to make them feel welcome.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
  return PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])


def together_ai_call(prompt):
  payload = {
      "model": TOGETHER_AI_MODEL,
      "prompt": prompt,
      "max_tokens": 2000,
      # "stop": ".",
      "temperature": 0.1,
      # "top_p": 0.7,
      # "top_k": 2,
      # "repetition_penalty": 1
  }
  headers = {
      "accept": "application/json",
      "content-type": "application/json",
      "Authorization": f"Bearer {API_KEY}"
  }
  response = requests.post(TOGETHER_AI_URL, json=payload, headers=headers)
  return response.json()


def retrieval_qa_chain(prompt, db, embeddings):

  def qa_chain(query):
    query_vector = embeddings.embed_documents([query])[0]
    retrieved_context = db.search(query, search_type='similarity', k=5)
    formatted_context = ' '.join(doc.page_content for doc in retrieved_context)
    formatted_prompt = prompt.format(context=formatted_context, question=query)
    return together_ai_call(formatted_prompt), retrieved_context

  return qa_chain


def qa_bot():
  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
      model_kwargs={'device': 'cpu'})
  db = FAISS.load_local(DB_FAISS_PATH, embeddings)
  qa_prompt = set_custom_prompt()
  return retrieval_qa_chain(qa_prompt, db, embeddings)


@cl.on_chat_start
async def start():
  chain = qa_bot()
  await cl.Message(content="Hi, Welcome to Medchat. What is your query?"
                   ).send()
  cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
  chain = cl.user_session.get("chain")
  api_response, sources = chain(message.content)

  # Extract the response text
  response_text = api_response.get('output', {}).get('choices', [{}])[0].get(
      'text', 'No response generated.').strip() if isinstance(
          api_response, dict) else "API response format is not as expected."

  # Send only the response text without appending source_info
  await cl.Message(content=response_text).send()
