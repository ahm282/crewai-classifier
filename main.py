#!/usr/bin/env python
from lib.pdf_reader import PdfReader
from lib.parse_xml import get_categories
from crew import classifier_crew


def train():
    """
    Train the crew for a given number of iterations.
    """
    pdf_reader = PdfReader()
    pdf_content = pdf_reader.read_upto_page(
        "./documents/GDP monthly estimate, UK September 2024.pdf", 10
    )

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


def run():
    """
    Run the crew on a given input.
    """
    # Read pdf content
    pdf_reader = PdfReader()
    pdf_content = pdf_reader.read_upto_page(
        "./documents/Lichfield_District_local_plan_strategy.pdf", 10
    )

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
    result = run()
    print(result)
