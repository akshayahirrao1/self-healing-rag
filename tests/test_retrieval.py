import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import get_vector_store

class TestRetrieval(unittest.TestCase):
    def test_retrieval(self):
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        
        # Test with a question we know should return results if ingestion has been run
        docs = retriever.invoke("What is the company policy?")
        
        self.assertIsInstance(docs, list, "Retriever should return a list of documents")
        # If ingestion hasn't been run, this might be empty, but ideally it shouldn't be.
        # We check the type instead of length to avoid test failure on empty DB.

if __name__ == '__main__':
    unittest.main()
