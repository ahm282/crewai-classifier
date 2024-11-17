#!/usr/bin/env python
from lib.pdf_reader import PdfReader
from lib.parse_xml import get_categories
from crew import classifier_crew


def train(pdf_file_path):
    """
    Train the crew for a given number of iterations.

    Args:
        pdf_file_path (str): Path to the pdf file.

    Returns:
        Saves the trained crew to a pkl file.
    """
    pdf_reader = PdfReader()
    pdf_content = pdf_reader.read_upto_page(pdf_file_path, 10)

    # Parse xml
    classifications = get_categories("./TOS/english_tos.xml")

    # Prepare inputs
    inputs_dict = {
        "raw_input": pdf_content,
        "hierarchy": classifications,
    }

    try:
        # Train the crew
        classifier_crew.train(
            n_iterations=10, filename="trained_crew.pkl", inputs=inputs_dict
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def run(pdf_file_path):
    """
    Run the crew on a given input.
    """
    # Read pdf content
    pdf_reader = PdfReader()
    pdf_content = pdf_reader.read_upto_page(pdf_file_path, 10)

    # Parse xml
    classifications = get_categories("./TOS/english_tos.xml")

    # Prepare inputs
    inputs_dict = {
        "raw_input": pdf_content,
        "hierarchy": classifications,
    }

    # Run the crew
    final_output = classifier_crew.kickoff(
        inputs=inputs_dict,
    )

    return final_output


if __name__ == "__main__":
    pdf_file = "./documents/Crime_statistics_in_California.pdf"
    result = run(pdf_file)
    print(result)
