import os
from langchain.document_loaders import UnstructuredFileLoader  # Loader to read various document types
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into manageable chunks
from langchain.embeddings import HuggingFaceEmbeddings  # Turns text into vector embeddings
from langchain.vectorstores import FAISS  # Vector database for similarity search
from transformers import pipeline  # Hugging Face pipeline to load summarization models

# -------------------- Step 1: Load Document --------------------

# Define the folder where your raw documents are stored
DATA_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\GENAITRAIN\GenAIvolve\GenAIwolf\data\raw"

FILENAME = "sample.pdf"  # Change this to match your document name

# Use LangChain's UnstructuredFileLoader to read the document
loader = UnstructuredFileLoader(os.path.join(DATA_PATH, FILENAME))

# Load the document into memory (as a list of Document objects)
documents = loader.load()

# -------------------- Step 2: Chunking --------------------

# Initialize the text splitter to break documents into smaller chunks
# Each chunk will be up to 1000 characters, with 200 characters overlapping between chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Apply the splitter to the loaded document(s)
docs = splitter.split_documents(documents)

# -------------------- Step 3: Embedding & Vector Store --------------------

# Initialize the Hugging Face embedding model (MiniLM - small, fast, effective for semantic search)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the document chunks and their embeddings
db = FAISS.from_documents(docs, embedding_function)

# -------------------- Step 4: Setup Retriever --------------------

# Turn the vector store into a retriever object
# This lets you search for relevant document chunks based on a user's question
retriever = db.as_retriever()

# -------------------- Step 5: Hugging Face Summarizer --------------------

# Load a Hugging Face summarization model (BART-large-cnn)
# This model can generate summaries from longer texts
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)


# -------------------- Step 6: Ask a Question --------------------

# Define the user’s query (you can change this to test different questions)
query = "What is this document about?"

# Use the retriever to find the most relevant document chunks for the query
relevant_docs = retriever.get_relevant_documents(query)

# Join the content of the retrieved chunks into a single string
context = "\n".join([doc.page_content for doc in relevant_docs])

# Print a preview of the retrieved context (up to 1000 characters)
print("\n🔎 Relevant context retrieved:\n", context[:1000])

# -------------------- Step 7: Summarize --------------------

# Use the summarizer to generate a summary of the retrieved context
# max_length and min_length control the length of the summary
summary = summarizer(context, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

# Print the summary
print("\n📝 Summary:\n", summary)
