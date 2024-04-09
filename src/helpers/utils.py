import os
import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from langchain.docstore.document import Document

class Utils:
  def __init__(self):
    pass

  #Retrieval utils
  def extract_booklet_number(self,filename: str) -> str:
      match = re.search(r'TG Booklet \d+', filename)
      if match:
          return match.group()
      else:
          return ""

  def split_content(self,content, max_words=1000):
      """
      Splits the content into chunks of approximately `max_words` words each.
      """
      content = str(content)
      words = content.split()
      num_words = len(words)
      num_chunks = (num_words + max_words - 1) // max_words  # Ceiling division to calculate number of chunks

      chunks = []
      for i in range(num_chunks):
          start_index = i * max_words
          end_index = min((i + 1) * max_words, num_words)
          chunk = " ".join(words[start_index:end_index])
          chunks.append(chunk)

      return chunks


  def process_csv(self,folder_path, csv_file):
      """
      Reads the CSV file, processes content, and creates Document objects.
      """
      documents = []
      booklet_number = self.extract_booklet_number(csv_file)
      if booklet_number:
          print(booklet_number)
          # Create a DataFrame for each file with the format book_i
          df = pd.read_excel(os.path.join(folder_path, csv_file))
          df.columns = ['paragraph', 'content']
          df['filename'] = booklet_number

          # Assuming you have a CSV reader (e.g., using pandas)
          for index, row in df.iterrows():
              content = row["content"]
              paragraph_number = row["paragraph"]
              filename = row['filename']

              if len(str(content).split()) > 1000:
                  # Split content into chunks
                  content_chunks = self.split_content(content, max_words=1000)
                  for i, chunk in enumerate(content_chunks):
                      doc = Document(
                          page_content=chunk,
                          metadata={
                              "source": filename,
                              "paragraph": paragraph_number,
                          }
                      )
                      documents.append(doc)
              else:
                  # No need to split
                  doc = Document(
                      page_content=content,
                      metadata={
                          "source": filename,
                          "paragraph": paragraph_number,
                      }
                  )
                  documents.append(doc)

      return documents

  def prepare_docs_list(self, folder_path: str):
      # Initialize an empty list to store DataFrames
      docs_all = []

      for file in os.listdir(folder_path):
          docs = self.process_csv(folder_path=folder_path, csv_file=file)
          docs_all.extend(docs)

      return docs_all

  def clean_whitespace(self, text):
      """Removes various types of excessive whitespace from a string."""
      try:
        # Replace multiple consecutive spaces with a single space
        text = re.sub(r"\s{2,}", " ", text)

        # Remove special whitespace characters like thin spaces, non-breaking spaces, etc.
        text = re.sub(r"[\u200B\u202F\u00A0]", " ", text)

        # Remove leading and trailing whitespace
        text = text.strip()

        return text
      except Exception as e:
        print(f"An exception occurred: {e}")
        return text



  def load_test_data(self, test_filepath: str):
      df_test = pd.read_csv(test_filepath)
      return df_test
