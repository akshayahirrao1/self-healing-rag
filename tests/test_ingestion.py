import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import load_documents, split_documents

class TestIngestion(unittest.TestCase):
    def test_load_documents(self):
        # We assume the data directory has at least company_policy.txt
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        
        docs = load_documents(data_dir)
        self.assertGreater(len(docs), 0, "Should load at least one document")
        
    def test_split_documents(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        docs = load_documents(data_dir)
        
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), len(docs), "Should split documents into multiple chunks")

if __name__ == '__main__':
    unittest.main()
