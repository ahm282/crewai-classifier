from lib.parse_xml import get_categories
from crewai import Task
from tools.chromadb_vectorizer import document_vectorizer, categories_vectorizer
from agents import (
    text_ingestion_specialist,
    text_preprocessing_specialist,
    text_classification_specialist,
    confidence_scoring_agent,
    human_in_the_loop,
)

# Define the hierarchy of classification labels
hierarchy = get_categories("./TOS/english_tos.xml")

# Define tasks with strict top-5 classification requirements
ingest_documents = Task(
    agent=text_ingestion_specialist,
    description="""
    Prepare and standardize the document and all incoming text {raw_input} by:
    1. Applying consistent formatting, encoding, and metadata structure
    2. Filtering out non-text content
    4. Validating that document content contains sufficient information for label matching
    """,
    expected_output="A structured and standardized document collection with metadata, verified for classification readiness.",
)

preprocess_text = Task(
    agent=text_preprocessing_specialist,
    description="""
    Prepare and vectorize document for semantic classification:
    1. Text Cleaning:
        - Remove noise (special characters, irrelevant symbols)
        - Correct spelling and formatting errors
        - Normalize whitespace and punctuation
    
    2. Semantic Processing:
        - Extract key semantic phrases matching hierarchy labels
        - Generate embeddings using ChromaDB vectorizer
        - Create semantic chunks for better matching
    
    3. Vector Enhancement:
        - Index document vectors in ChromaDB
        - Compute semantic similarity with label descriptions
        - Generate relevance scores for potential labels

    Ensure vectors are properly indexed for subsequent label matching.
    """,
    expected_output="Preprocessed document with ChromaDB vectors and semantic similarity scores",
    context=[ingest_documents],
)

classify_documents = Task(
    agent=text_classification_specialist,
    description="""
    Perform strict top-5 label classification by:
    1. Analyzing the content of the document against the provided {hierarchy}.
    2. Using the `document_vectorizer` tool to generate a vector representation of the document.
    3. Using the `categories_vectorizer` tool to generate vector embeddings for each label in the hierarchy.
    4. Computing similarity scores between the document vector and the category vectors.
    5. STRICTLY selecting only the top 5 highest-scoring labels based on similarity.
    6. Recording detailed justification for each selected label, including:
        - Similarity score for each label.
        - Key evidence supporting each label selection.
    7. Ensuring exactly 5 labels are chosen - no more, no less.

    Output:
    - Top 5 labels in ranked order.
    - Similarity score for each label.
    - Key evidence supporting each label selection.
    
    Include a structured output in JSON format:
        {{
            "validated_classifications": [
                {{"label": "Label1", "confidence_score": 0.85, "Reasoning": "Reasoning"}},
                {{"label": "Label2", "confidence_score": 0.75, "Reasoning": "Reasoning"}},
                {{"label": "Label3", "confidence_score": 0.70, "Reasoning": "Reasoning"}},
                {{"label": "Label4", "confidence_score": 0.68, "Reasoning": "Reasoning"}},
                {{"label": "Label5", "confidence_score": 0.65, "Reasoning": "Reasoning"}}
            ],
            "improvement_feedback": "Suggestions for improving classification accuracy."
        }}

    Ensure that the output strictly follows the above JSON structure.
    """,
    expected_output="""Documents with exactly 5 ranked labels, scores, and supporting evidence.
    The output should be a JSON object with the following structure:
        {{
            "validated_classifications": [
                {{"label": "Label1", "confidence_score": 0.85, "Reasoning": "Reasoning"}},
                {{"label": "Label2", "confidence_score": 0.75, "Reasoning": "Reasoning"}},
                {{"label": "Label3", "confidence_score": 0.70, "Reasoning": "Reasoning"}}
            ],
            "improvement_feedback": "Suggestions for improving classification accuracy."
        }}
    """,
    context=[preprocess_text],
    allowed_labels=hierarchy,
    tools=[document_vectorizer, categories_vectorizer],
)

score_confidence = Task(
    agent=confidence_scoring_agent,
    description="""
    Evaluate classification confidence for top-5 labels by:
    1. Analyzing strength of evidence for each assigned label
    2. Computing confidence scores (0-1) for each label
    3. Flagging cases where:
        - Gap between 2nd, 3rd, 4th or 5th best label is small (<0.1)
        - Confidence score for any top-5 label is below 0.6
        - Evidence for labels is contradictory
    """,
    expected_output="""Confidence scores for each top-5 label with specific flag conditions documented.
        The output should is a JSON object with strcitly the following structure:
        {{
            "validated_classifications": [
                {{"label": "Label1", "confidence_score": 0.85, "Reasoning": "Reasoning"}},
                {{"label": "Label2", "confidence_score": 0.75, "Reasoning": "Reasoning"}},
                {{"label": "Label3", "confidence_score": 0.70, "Reasoning": "Reasoning"}}
            ],
            "improvement_feedback": "Suggestions for improving classification accuracy"
        }}
    """,
    context=[preprocess_text, classify_documents],
    allowed_labels=hierarchy,
)

human_review = Task(
    agent=human_in_the_loop,
    description="""
    Review flagged classifications by:
    1. Examining documents with low confidence scores or flags
    2. Validating or adjusting the top-5 label selections
    3. Maintaining strict adherence to 5-label requirement
    4. Documenting rationale for any label changes
    5. Providing feedback to improve automated classification
    You are to adhere to the expected output format. Only JSON format is accepted.
    No more text or other formats are allowed. We are very strict on this.
    Ensure output maintains exactly 5 labels per document strictly from {hierarchy}.
    """,
    context=[classify_documents, score_confidence],
    expected_output=""" Your output should be in the following format and this is the only format accepted.
    No additional text or formatting is allowed. You do not get to decide the format.:
        {{
            "validated_classifications": [
                {{"label": "Label1", "confidence_score": 0.85, "Reasoning": "Reasoning"}},
                {{"label": "Label2", "confidence_score": 0.75, "Reasoning": "Reasoning"}},
                {{"label": "Label3", "confidence_score": 0.70, "Reasoning": "Reasoning"}}
            ],
            "improvement_feedback": "Suggestions for improving classification accuracy"
        }}
    """,
)
