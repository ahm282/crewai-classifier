# crewai-classifier

# CrewAI Document Classifier

CrewAI Document classifier is a tool that uses machine learning to categorize documents into predefined classes. It supports various document formats and can be integrated into existing workflows.

## Installation

To install the CrewAI Document Classifier, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/crewai-classifier.git
    ```
2. Navigate to the project directory:
    ```sh
    cd crewai-classifier
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use the CrewAI Document Classifier, follow these steps:

1. Prepare your documents in the supported formats (`.pdf` only currently).
2. Change the file path in `main.py`:

    ```sh
        classifications = get_categories("./TOS/english_tos.xml")

        inputs_dict = {
            "raw_input": pdf_content,
            "hierarchy": classifications,
        }
    ```

3. Run the classifier:
    ```sh
    python main.py
    ```

## To-do list

-   [ ] Add persistent ChromaDB or any other vectordb
-   [ ] Implement NLP
-   [ ] Experiment with different LLMs

## Contributing

We welcome contributions to improve the CrewAI Document Classifier. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```sh
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```sh
    git commit -m "Description of changes"
    ```
4. Push to the branch:
    ```sh
    git push origin feature-branch
    ```
5. Create a pull request.
