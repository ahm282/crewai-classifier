import pdfplumber


class PdfReader:
    def __init__(self):
        pass

    def open_pdf(self, file_path: str) -> pdfplumber.PDF:
        """
        Open a PDF file using pdfplumber.

        Args:
            pdf_file (str): Path to the PDF file

        Returns:
            pdfplumber.PDF: PDF object
        """
        try:
            return pdfplumber.open(file_path)
        except Exception as e:
            raise Exception(f"An error occurred while opening the PDF file: {e}")

    def close_pdf(self, pdf_file: pdfplumber.PDF):
        """
        Close the PDF file.

        Args:
            pdf_file (pdfplumber.PDF): PDF object
        """
        try:
            pdf_file.close()
        except Exception as e:
            raise Exception(f"An error occurred while closing the PDF file: {e}")

    def read_all_pages(self, file_path: str) -> list:
        """
        Read all pages of a PDF file using pdfplumber. Return the content as a list of strings.

        Args:
            pdf_file (str): Path to the PDF file

        Returns:
            list: Content of the PDF file
        """

        pdf_file = pdfplumber.open(file_path)
        content = []

        try:
            pdf = self.open_pdf(file_path)

            for page in pdf.pages:
                content.append(page.extract_text())
            return content
        except Exception as e:
            self.close_pdf(pdf_file)
            raise Exception(f"An error occurred while reading the PDF file: {e}")

    def read_page(self, file_path: str, page_number: int) -> str:
        """
        Read a specific page of a PDF file using pdfplumber.
        If page_number is too high, read the last page.

        Args:
            pdf_file (str): Path to the PDF file
            page_number (int): Page number to read

        Returns:
            str: Content of the PDF file
        """

        pdf_file = pdfplumber.open(file_path)

        try:
            if page_number > len(pdf_file.pages):
                page_number = len(pdf_file.pages) - 1

            self.close_pdf(pdf_file)
            return pdf_file.pages[page_number].extract_text()
        except Exception as e:
            self.close_pdf(pdf_file)
            raise Exception(f"An error occurred while reading the PDF file: {e}")

    def read_upto_page(self, file_path: str, page_number: int) -> list:
        """
        Read all pages of a PDF file up to a specific page number using pdfplumber.
        Return the content as a list of strings.

        Args:
            pdf_file (str): Path to the PDF file
            page_number (int): Page number to read up to

        Returns:
            list: Content of the PDF file
        """

        pdf_file = pdfplumber.open(file_path)

        content = []

        try:
            if page_number > len(pdf_file.pages):
                page_number = len(pdf_file.pages) - 1

            for i in range(page_number):
                content.append(pdf_file.pages[i].extract_text())

            self.close_pdf(pdf_file)
            return content
        except Exception as e:
            self.close_pdf(pdf_file)
            raise Exception(f"An error occurred while reading the PDF file: {e}")

    def read_from_page(self, file_path: str, page_number: int) -> list:
        """
        Read all pages of a PDF file from a specific page number using pdfplumber.
        Return the content as a list of strings.

        Args:
            pdf_file (str): Path to the PDF file
            page_number (int): Page number to read from

        Returns:
            list: Content of the PDF file
        """

        pdf_file = pdfplumber.open(file_path)

        content = []

        try:
            if page_number > len(pdf_file.pages):
                page_number = len(pdf_file.pages) - 1

            for i in range(page_number, len(pdf_file.pages)):
                content.append(pdf_file.pages[i].extract_text())

            self.close_pdf(pdf_file)
            return content
        except Exception as e:
            self.close_pdf(pdf_file)
            raise Exception(f"An error occurred while reading the PDF file: {e}")
