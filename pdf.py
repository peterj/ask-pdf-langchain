from typing import Optional
from langchain.document_loaders import PyPDFLoader
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.openai_functions.base import create_structured_output_runnable

class Answer(BaseModel):
    short_summary: str = Field(..., description="Short summary of the answer")
    detailed_explanation: str = Field(..., description="Detailed explanation of the answer")
    page_number: int = Field(..., description="Page number where the answer was found")
    chapter_title: str = Field(..., description="Chapter title where the answer was found")
    author: Optional[str] = Field(..., description="Author of the book")

def get_pdf_pages(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

## TODO: Make sure book.pdf or other PDF file exists in the same folder.
pdf_pages = get_pdf_pages('book.pdf')
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(pdf_pages)

embeddings = OpenAIEmbeddings()

faiss_index_file = 'advanced_react_faiss_index'

# Check if the folder exists
if not os.path.exists(faiss_index_file):
    print('Creating FAISS index')
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(faiss_index_file)
else:
    print('Loading FAISS index')
    db = FAISS.load_local(faiss_index_file, embeddings)


retriever = db.as_retriever()

template = """Answer the question based only on the following context. Include a short summary, a long explanation, page number, chapter title and author name as a reference. If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Context:
{context}

Question: {question}

"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()


llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
formatting_prompt = ChatPromptTemplate.from_messages( [
        (
            "system",
            "You are a world class algorithm for extracting information in structured formats.",
        ),
        (
            "human",
            "Use the given format to extract information from the following input: {input}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ])

runnable_formatting = create_structured_output_runnable(Answer, llm, formatting_prompt)

chain_1 = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain_1_response = chain_1.invoke("How to use useLayoutEffect?")
result = runnable_formatting.invoke({"input": chain_1_response})
print(result)

# Uncomment this portion and replace it with the one above to use Mistral instead of OpenAI for the last step
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage


# api_key = os.environ.get('MISTRAL_API_KEY')

# if api_key is None:
#     print('MISTRAL_API_KEY environment variable not set')
#     exit(1)
# mistral_model = "mistral-tiny"
# client = MistralClient(api_key=api_key)

# messages = [
#     ChatMessage(role="system", content="You are a world class algorithm for extracting information in structured formats."),
#     ChatMessage(role="user", content=f"Use the JSON format with the following fields - short_summary, detailed_summary, page_number, chapter_title, author_name - and extract information from the following input: {chain_1_response}"),
#     ChatMessage(role="user", content="Tip: Make sure to answer in the correct format"),
# ]

# chat_response = client.chat(
#     model=mistral_model,
#     messages=messages,
# )
# print(chat_response.choices[0].message.content)






# query = 'How to use useLayoutEffect?'

# result_docs= db.similarity_search_with_score(query)

# # Sort result_docs by scores
# result_docs = sorted(result_docs, key=itemgetter(1), reverse=True)


# print(result_docs[0])