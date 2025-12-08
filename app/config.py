# app/config.py

import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "rag_embeddings")

# Extract DB name from URI — last part after the last slash
DB_NAME = MONGODB_URI.rsplit("/", 1)[-1]

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

# use safe variable name
docs_collection = db[MONGODB_COLLECTION]
