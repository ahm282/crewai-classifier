import os
from crewai import Crew, Process
import datetime
from tasks import (
    ingest_documents,
    preprocess_text,
    classify_documents,
    score_confidence,
    human_review,
)

# Set environment variables if required
os.environ["OPENAI_API_KEY"] = "sk-1234567890abcdef1234567890abcdef"

# Set the current date and time for logging
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create and configure the crew
classifier_crew = Crew(
    agents=[
        ingest_documents.agent,
        preprocess_text.agent,
        classify_documents.agent,
        score_confidence.agent,
        human_review.agent,
    ],
    tasks=[
        ingest_documents,
        preprocess_text,
        classify_documents,
        score_confidence,
        human_review,
    ],
    process=Process.sequential,
    output_log_file=f"./crew_output.log-{timestamp}",
    share_crew=False,
    verbose=True,
    embedder={"provider": "ollama", "config": {"model": "nomic-embed-text:latest"}},
)
