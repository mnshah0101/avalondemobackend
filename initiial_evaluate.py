from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from datasets import Dataset
from llama_index.vector_stores.pinecone import PineconeVectorStore as PineconeVectorStoreLlama
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import OpenAIEmbeddings
from pinecone.grpc import PineconeGRPC
from langchain_community.document_loaders import PyPDFLoader
from ragas import evaluate
from rag import get_rachel_answer
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from dotenv import load_dotenv
import os

from langfuse import Langfuse
langfuse = Langfuse(
    secret_key="",
    public_key="",
    host="https://cloud.langfuse.com"
)

load_dotenv()

MONGO_URI = os.environ.get('MONGO_URI')

# Configuration and initialization


def initialize_mongo_client(uri):
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)
        return None

# Generate testset


def generate_testset(docs, doc_length):
    generator = TestsetGenerator.with_openai()
    testset = generator.generate_with_langchain_docs(docs[:int(
        doc_length/2)], test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
    return testset

# Create dataset


def create_dataset(df, pinecone_index):
    pc = PineconeGRPC()
    # df = testset.to_pandas()
    questions = df["question"].tolist()
    ground_truth = df["ground_truth"].tolist()
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=pinecone_index, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    data = {
        "question": questions,
        "answer": [get_rachel_answer(query, pinecone_index) for query in questions],
        "contexts": [[doc.page_content for doc in retriever.get_relevant_documents(query)] for query in questions],
        "ground_truth": ground_truth
    }
    return Dataset.from_dict(data)

# Evaluate dataset


def evaluate_dataset(dataset):
    result = evaluate(
        dataset=dataset,
        metrics=[context_relevancy, context_precision,
                 context_recall, faithfulness, answer_relevancy],
    )
    return result.to_pandas()

# Logging


def log_questions(df, collection):
    data_dict = df.to_dict("records")
    for record in data_dict:
        collection.insert_one(record)


def log_model_performance(df, model, model_description, collection):
    averages = {metric: df[metric].mean() for metric in [
        'context_relevancy', 'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']}
    document = {"model_name": model,
                "model_description": model_description, **averages}
    result = collection.insert_one(document)
    print(f"Document inserted with id {result.inserted_id}")

# Example of usage
# uri = "mongodb+srv://username:password@yourcluster.mongodb.net/?retryWrites=true&w=majority&appName=model-tracker"
# client = initialize_mongo_client(uri)
# db = client['model-tracker']
# models_collection = db['models']
# questions_collection = db['questions']

# docs = [...]  # Assume docs is already defined
# doc_length = len(docs)
# testset = generate_testset(docs, doc_length)
# dataset = create_dataset(testset)
# results_df = evaluate_dataset(dataset)

# log_questions(results_df, questions_collection)
# log_model_performance(results_df, "Model_Name", "Description of the model", models_collection)


# documents = load your documents

# generator with openai models


def generate_testset2(docs):
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    # Change resulting question type distribution
    distributions = {
        simple: 0.5,
        multi_context: 0.4,
        reasoning: 0.1
    }

    # use generator.generate_with_llamaindex_docs if you use llama-index as document loader
    testset = generator.generate_with_langchain_docs(docs, 10, distributions)
    testset.to_pandas()
    return testset


def load_pdfs():
    loaders = []
    for filename in os.listdir('./avalonpdfs'):
        try:
            loaders.append(PyPDFLoader(
                f'avalonpdfs/{filename}', headers={"document_link": "hi"}))
        except Exception as e:
            continue

    docs = []
    for l in loaders:
        docs.extend(l.load())
    return docs

# def evaluate(pinecone_index, docs, description):
#     client = initialize_mongo_client(MONGO_URI)
#     db = client['model-tracker']
#     models_collection = db['models']
#     questions_collection = db['questions']
#     print(questions_collection)

#     doc_length = len(docs)
#     testset = generate_testset2(docs)
#     print(testset)
#     dataset = create_dataset(testset, pinecone_index)
#     results_df = evaluate_dataset(dataset)

#     log_questions(results_df, questions_collection)
#     log_model_performance(results_df, "Model_Name", description, models_collection)

#     return results_df


def evaluate2(pinecone_index, testset, description):
    client = initialize_mongo_client(MONGO_URI)
    db = client['model-tracker']
    models_collection = db['models']
    questions_collection = db['questions']
    dataset = create_dataset(testset, pinecone_index)
    results_df = evaluate_dataset(dataset)

    trace = langfuse.trace(
        name=f"{description}",
        user_id="admin",
        metadata={
            "email": "prod@company.com",
        },
        tags=["evaluation"]
    )
    for _, row in results_df.iterrows():
        for metric_name in ["faithfulness", "answer_relevancy", "context_recall"]:
            langfuse.score(
                name=metric_name,
                value=row[metric_name],
                trace_id=trace.id
            )

    # log_questions(results_df, questions_collection)
    # log_model_performance(results_df, "Model_Name", description, models_collection)
    return results_df
