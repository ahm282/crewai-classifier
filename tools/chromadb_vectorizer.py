from chromadb import Client
import chromadb.utils.embedding_functions as embedding_functions
from crewai_tools import tool
from lib.parse_xml import get_categories

# Initialize ChromaDB client
client = Client()
collection = client.get_or_create_collection("document_embeddings")
vectorizer_model = "nomic-embed-text:latest"

# Initialize Ollama embedding function
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name=vectorizer_model,
    url="http://localhost:11434/v1",
)


@tool
def document_vectorizer(text: str) -> str:
    """
    Vectorize the input text and store the embedding in ChromaDB.

    Args:
    text (str): The input text to be vectorized.

    Returns:
    str: A message indicating the successful vectorization and storage of the embedding.
    """
    # Store the embedding in ChromaDB collection
    embeddings = ollama_ef(text)
    collection.add(documents=[text], ids=[text], embeddings=[embeddings])
    return "Document vectorized and added to ChromaDB."


@tool
def categories_vectorizer():
    """
    Vectorize the categories and store the embeddings in ChromaDB.

    Returns:
        str: A message indicating the successful vectorization and storage of the category embeddings.
    """
    # Get the categories from the XML file (or predefined dictionary)
    categories = get_categories("../TOS/english_tos.xml")

    # Extract the descriptions for vectorization
    category_descriptions = list(
        categories.values()
    )  # Extract the category text descriptions

    # Get embeddings for the category descriptions
    embeddings = ollama_ef(category_descriptions)

    collection.add(
        documents=category_descriptions,
        ids=list(categories.keys()),
        embeddings=embeddings,
    )

    return "Categories vectorized and added to ChromaDB."
