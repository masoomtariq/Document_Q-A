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
from langchain.retrievers.multi_query import MultiQueryRetriever # For creating a retriever with multiple queries
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage # For creating messages
from operator import itemgetter # For accessing items in dictionaries
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import START, END, StateGraph, MessagesState, add_messages
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import os

## Langsmith tracking (for experiment tracking, if you have an account)
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_PROJECT']  = "Document Q/A"
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]

groq_key = st.secrets["GROQ_API_KEY"]
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
        return vectors.as_retriever()
    except Exception as e:
        st.error(f"Error creating vector store: {e}") # Display error
        return None
    
#groq_1 = ChatGroq(api_key=groq_key, temperature=0, model="llama-3.2-1b-preview")
#groq_3 = ChatGroq(api_key=groq_key, temperature=0, model="llama-3.2-3b-preview")
#groq_11 = ChatGroq(api_key=groq_key, temperature=0, model="llama-3.2-11b-vision-preview")
groq_90 = ChatGroq(api_key=groq_key, temperature=0, model="llama-3.2-90b-vision-preview")

google = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=google_key, temperature=0)
google1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", api_key=google_key, temperature=0)
google2 = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-lite-preview-02-05", api_key=google_key, temperature=0)

def get_summary():
    text = ''
    for i in st.session_state.docs:
        text += i.page_content.strip()
        text += "\n"
    response = google.invoke(f"""Act like a text Summarizer.
    Instruction: You will the summarize the text efficiently then that summary will be used in Analyzing the query.
    So keep in mind that Instructions and Summarize the following text:

    Text: {text}""")
    return response.content

# Creates the vector store and processes the uploaded file
def vectorize():
    with st.status("Processing..."):
        if 'retriever' not in st.session_state:
            st.toast("Document Loading...")
            st.session_state.docs = load_document()
            clean_pg_meta()

            st.toast("Document Splitting...")
            splitt = splitter(chunk_size = 1000, chunk_overlap = 100, separators=['\n\n', '.','\n', ' ', ''])
            st.session_state.splits = splitt.split_documents(st.session_state.docs)
            clean_chunk_meta()

            st.session_state.retriever = vectore_store()
            st.toast("Document Uploaded", icon='🎉')

class state(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    classification_result: str
    response: str

def query_classifier(state):
    query = state['messages'][-1].content
    template = f"""You are an intelligent query analyzer.
        Your task is to Search through the DataBase's Summary then Analyze that Is there's Any Meaning in the user's query is relevent and Return Only True or False
        Instructions:
        - Understand the Query and identify if there are any keywords, meaningful words, or concepts that are related to DataBase.
        - If query is scrambled then first transform the query and then analyze the query
        - Determine the relevency Score of qeury and DataBase's Summary
        - Return True If the relevency Score is greater than 0.5
        - Otherwise return False

        Keep In mind Return Only True Or False

        DataBase's
        Summary : {st.session_state.summary}

        Query : {query}"""

    response = groq_90.invoke(template)
    classification_result = response.content.strip().lower()
    return {'classification_result': classification_result}

def retrieve(state):
    sysmsg = "You are a document Q/A assistant you task is to answer the user's questions based the given context/n<context>\n{context}\n<context>"
    prompt = ChatPromptTemplate.from_messages(messages=[('system', sysmsg),
                                                        ("placeholder", "{messages}"),
                                                        ('human', "{query}")])
    query = state['messages'][-1].content
    set_up = {'query': itemgetter('query'), 'context': itemgetter('query') | st.session_state.multi_retriever}
    chain = set_up | prompt | google1
    response = chain.invoke({'messages': state['messages'][:-1], 'query': query})
    return {'messages': response, 'response': response}

def assistant(state):
    llm = google2.bind_tools([st.session_state.retriever_tool])
    response = llm.invoke(state['messages'])
    return {'messages': response, 'response': response}

def building_graph():
    workflow = StateGraph(state)
    workflow.add_node("router", query_classifier)
    workflow.add_node("retriever", retrieve)
    workflow.add_node("assistant", assistant)
    workflow.add_node("tools", ToolNode(tools=[st.session_state.retriever_tool]))

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges('router', lambda state:state['classification_result'], {'true': 'retriever', 'false': 'assistant'})
    workflow.add_conditional_edges("assistant", tools_condition)
    workflow.add_edge("tools", 'assistant')

    workflow = workflow.compile(checkpointer=MemorySaver())

    return workflow

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
    # Specify a thread
    config = {"configurable": {"thread_id": "1"}}

    # Specify an input
    messages = st.session_state.messages+[HumanMessage(content=st.session_state.entered_prompt)]
    workflow = building_graph()
    response = workflow.invoke({"messages": messages}, config)
    st.session_state.messages = response['messages']
    st.session_state['past'].append(st.session_state.entered_prompt)
    st.session_state['generated'].append(response['response'].content)

def initialize_state():
    # Define initial states and assign values
    initialStates = {
        'generated': [],
        'past': [],
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

                st.session_state.summary = get_summary()
                st.session_state.multi_retriever = MultiQueryRetriever.from_llm(retriever=st.session_state.retriever, llm=google2, include_original=True)
                
                tool_name = st.session_state.collection_name.replace(' ', '_')+'_Search'
                description = f"Search for the relevant information in the '{st.session_state.collection_name}' document"
                st.session_state.retriever_tool = st.session_state.retriever.as_tool(name=tool_name, description=description)


            st.divider()

        if 'retriever' in st.session_state:
                if st.button("End Session", key="clear", help="Clear the file"):
                    clear_session()
                    st.success("Session_State is Cleared. Document is removed")

    if 'retriever' in st.session_state:
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
