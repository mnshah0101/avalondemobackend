import boto3
from pinecone import ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import S3FileLoader
from pinecone.grpc import PineconeGRPC
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from supabase import create_client
from dotenv import load_dotenv
from gpt import getDocSummary
import os

load_dotenv()

url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')

supabase = create_client(url, key)


def load_s3_files(bucket_name, doc_names):
    try:
        docs = []
        for doc_name in doc_names:
            print(doc_name)
            loader = S3FileLoader(bucket_name, doc_name)
            doc = loader.load()
            docs.append([doc, doc_name])
        return docs
    except Exception as e:
        print(f"Error : {e}")


def upload_to_pinecone(bucket_name: str, username: str, doc_names: list):
    try:
        # load documents from s3
        docs = load_s3_files(bucket_name, doc_names)

        # get summaries for each doc
        doc_summaries = getDocSummary(docs)

        # update supabase table if case does not already exist
        case, _ = supabase.table('cases').select(
            '*').eq('case_name', bucket_name).execute()
        print(case)
        if not case[1]:
            supabase.table('cases').insert(
                {'case_name': bucket_name, 'user': username}).execute()

        # upload to pinecone
        pc = PineconeGRPC()
        index_name = bucket_name
        indices = pc.list_indexes()
        index = None
        for existing_index in indices:
            if index_name == existing_index['name']:
                index = pc.Index(index_name)
                print(f'found existing index {bucket_name}')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256, chunk_overlap=20)
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        if index:
            for doc, name in docs:
                chunks = text_splitter.split_documents(doc)
                print(f'updating {name} to index {bucket_name}')
                PineconeVectorStore.from_documents(
                    chunks, embeddings, index_name=bucket_name)

                # update supabase table
                object_url = f"https://{bucket_name}.s3.amazonaws.com/{name}"
                supabase.table('files').insert(
                    {'file': name, 'url': object_url, 'case': bucket_name, "user": username, "summary": doc_summaries[name]}).execute()
            print("Success")
        else:
            print(f'creating new index {bucket_name}')
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2",
                )
            )

            for doc, name in docs:
                chunks = text_splitter.split_documents(doc)
                print(f'updating {name} to index {bucket_name}')
                PineconeVectorStore.from_documents(
                    chunks, embeddings, index_name=bucket_name)

                # update supabase table
                object_url = f"https://{bucket_name}.s3.amazonaws.com/{name}"
                supabase.table('files').insert(
                    {'file': name, 'url': object_url, 'case': bucket_name, "user": username, "summary": doc_summaries[name]}).execute()
            print("Success")
    except Exception as e:
        print(f"Error : {e}")


def delete_from_pinecone(docs: list, bucket_name: str):
    try:
        pc = PineconeGRPC()
        index_name = bucket_name
        indices = pc.list_indexes()
        index = None
        for existing_index in indices:
            if index_name == existing_index['name']:
                index = pc.Index(index_name)
                print(f'found existing index {bucket_name}')

        # delete from pinecone and supabase
        if index:
            for doc in docs:
                index.delete(ids=[doc])
                doc, _ = supabase.table('files').delete().eq(
                    'file', doc).execute()
            print("Success")
        else:
            print(f'Index {bucket_name} not found')
    except Exception as e:
        print(f"Error : {e}")
