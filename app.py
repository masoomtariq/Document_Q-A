import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter as splitter # For splitting text into chunks
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings #For creating text embeddings
from langchain_core.vectorstores import InMemoryVectorStore # For storing and searching vectors
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI # For using Gemini LLM
from langchain.prompts import ChatPromptTemplate # For creating prompt templates
from langchain.load import dumps, loads
from operator import itemgetter # For accessing items in dictionaries
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
import os

## Langsmith tracking (for experiment tracking, if you have an account)
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_PROJECT']  = "Document Q/A"
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]

#groq_key = st.secrets["GROQ_API_KEY"]
google_key = st.secrets["GOOGLE_API_KEY"]

# Loads a document from the given file path, handling different file types
def load_document():
    # Mapping of file extensions to loaders
    loaders = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
    }
    # Get the appropriate loader
    loader = loaders.get(st.session_state.file_ext)
    
    return loader(st.session_state.file_path).load() # Load the document
    
# Cleans metadata from loaded pages, adding source and page number
def clean_pg_meta():
    for index, page in enumerate(st.session_state.docs):
        page.metadata.clear() #Clear existing metadata
        page.metadata['source'] = st.session_state.file_source # Adding the source
        page.metadata['page_no'] = index + 1 # Adding the page_no
        page.page_content = page.page_content.replace('\n', ' ').strip() # Clean page content
# Cleans metadata from text chunks, adding chunk number
def clean_chunk_meta():
    for index, chunk in enumerate(st.session_state.splits):
        chunk.metadata['chunk_no'] = index + 1 # Add chunk number
        chunk.page_content = chunk.page_content.replace('\n', ' ').strip()

def vectore_store():
    embed=GoogleGenerativeAIEmbeddings(google_api_key=google_key, model="models/embedding-001")
    try:
        vectors = InMemoryVectorStore.from_documents(documents=st.session_state.splits, embedding=embed) # Create vector store
        return vectors
    except Exception as e:
        st.error(f"Error creating vector store: {e}") # Display error
        return None
    
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", temperature=0, api_key=google_key)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=google_key, temperature=0)

# Gets unique documents from a list of lists
def get_unique(docs : list[list]):
    flattened = [dumps(item) for sublist in docs for item in sublist]

    unique = list(set(flattened))
    return [loads(item) for item in unique]

def get_queries(query):
    template =f"""You are an AI language model assistant. Your task is to generate four
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    please do not include extra text do your best it is very important to my career.
    Original question: {query}"""
    response = model.invoke(template)
    queries = [query]+(lambda x : [i for i in x.split('\n') if i])(response.content)
    return queries

# Creates the vector store and processes the uploaded file
def vectorize():
    with st.status("Processing..."):
        if 'vectors' not in st.session_state:
            st.toast("Document Loading...")
            st.session_state.docs = load_document()
            clean_pg_meta()

            st.toast("Document Splitting...")
            splitt = splitter(chunk_size = 1000, chunk_overlap = 100, separators=['\n\n', '.','\n', ' ', ''])
            st.session_state.splits = splitt.split_documents(st.session_state.docs)
            clean_chunk_meta()

            st.session_state.vectors = vectore_store()
            st.toast("Document Uploaded", icon='🎉')

# Clears the current session, deleting the vector store and uploaded file
def clear_session():
    st.session_state.clear()
    st.toast("Session_State is Cleared")

def clear_chat():
    keys = ['generated', 'past', 'entered_prompt', 'messages']
    for i in keys:
        del st.session_state[i]
    st.rerun()

def generate_responce():

    retriever = st.session_state.vectors.as_retriever()
    retriever.search_kwargs['k'] = 3
    
    retreival = get_queries | retriever.map() | get_unique

    st.session_state.messages.append(("human", st.session_state.entered_prompt))
    prompt = ChatPromptTemplate.from_messages(messages = st.session_state.messages)

    chain = {'context': itemgetter('query') | retreival } | prompt | llm
    response = chain.invoke({'query': st.session_state.entered_prompt})

    st.session_state.messages.append(("ai", response.content))

    st.session_state['past'].append(st.session_state.entered_prompt)
    st.session_state['generated'].append(response.content)

def initialize_state():
    # Define initial states and assign values
    initialStates = {
        'generated': [],
        'past': [],
        'messages': [],
        'entered_prompt': '',
        'file': '',
        'file_path': '',
        'file_ext': '',
        'file_source': '',
        'collection_name': ''
    }
    for key, value in initialStates.items():
        if key not in st.session_state:
            st.session_state[key] = value

def initialize_chat():
    template = """
    You are a helpful and informative document question-answering assistant.  Your primary goal is to provide accurate and insightful answers based *exclusively* on the provided context.  You are an expert at synthesizing information and drawing connections within the given text.  Do not rely on any external knowledge or information beyond what is explicitly given in the context.
    If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
    Always maintain a polite and professional tone.
    **Instructions for Enhancing Context-Based Responses:**

    1. **Context is King:**  Treat the provided context as the absolute source of truth.  Base your entire response on this information.  If the context doesn't contain the answer, explicitly state that "The answer cannot be found within the provided context."  Do not hallucinate or make assumptions.

    2. **Deep Understanding:**  Carefully analyze the context to understand the nuances of the information presented.  Identify key concepts, relationships, and any implicit information conveyed.

    3. **Synthesis and Summarization:**  If the answer requires combining information from multiple parts of the context, synthesize the relevant pieces into a coherent and comprehensive response.  Summarize the key points concisely and accurately.

    4. **Clarity and Conciseness:**  Provide clear and concise answers.  Avoid unnecessary jargon or overly complex language.  Structure your response logically and use bullet points or numbered lists if appropriate to enhance readability.

    5. **Evidence-Based Answers:**  Whenever possible, directly quote or paraphrase specific sentences or phrases from the context to support your answer.  This demonstrates that your response is grounded in the provided information.  If you paraphrase, ensure you maintain the original meaning. At the end of the answer, Cite the page of the context that refer your answer.

    6. **Address the Question Directly:**  Make sure your answer directly addresses the question being asked.  Avoid going off on tangents or providing irrelevant information.

    7. **Handle Ambiguity:**  If the question is ambiguous or can be interpreted in multiple ways, acknowledge the ambiguity and provide possible answers based on different interpretations of the question, all within the bounds of the provided context.

    8. **Iterative Refinement:**  If you are unsure about the answer, re-read the context carefully and try to identify any clues or connections you may have missed.

    **Remember:** Your focus should be on extracting and synthesizing information *exclusively* from the provided context.  Your success depends on your ability to understand and apply these instructions.

    <context>
    {context}
    <context>
    """
    st.session_state.messages.append(("system", template))
    st.session_state.messages.append(("ai", "Hello! How can I help you today?"))
    message("Hello! How can I help you today?")

def update_file():
    st.session_state.file_source = st.session_state.file.name
    st.session_state.collection_name, ext = st.session_state.file_source.split('.')
    st.session_state.file_ext = '.'+ext
    # Save the uploaded file to the tempfile filesystem
    with tempfile.NamedTemporaryFile(delete=False, suffix=st.session_state.file_ext) as temp_file:

        temp_file.write(st.session_state.file.read())
        st.session_state.file_path = temp_file.name

def display_chat():
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=f"{str(i)}_user")
        message(st.session_state['generated'][i], key=str(i))

def main():

    st.set_page_config(page_title="Q/A", page_icon=":books:")
    st.title("Document Q/A :book:")

    initialize_state()

    # Create a form for uploading files
    with st.sidebar:
        st.session_state.file = st.file_uploader("Choose a file", help="Upload a file")
        
        if st.session_state.file is not None:

            if st.button("Submit", key="submit", help="Submit the file"): # If the submit button is clicked
                update_file()
                vectorize()

            st.divider()

        if 'vectors' in st.session_state:
                if st.button("End Session", key="clear", help="Clear the file"):
                    clear_session()
                    st.success("Session_State is Cleared. Document is removed")

    if 'vectors' in st.session_state:
        st.subheader(body=st.session_state.collection_name)
        # Define submit function and input field
        query = st.chat_input("Enter Prompt")
        # Check if 'entered_prompt' is empty or not
        if query:
            st.session_state.entered_prompt = query
            generate_responce()
            
        # Display Messages
        if st.session_state['generated'] == []:
            initialize_chat()
        if st.session_state['generated']:
            if st.button("Clear_Chat", help="Clear the chats"):
                clear_chat()
            display_chat()
    

if __name__ == "__main__":
    main()
