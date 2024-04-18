from pinecone.grpc import PineconeGRPC
from llama_index.vector_stores.pinecone import PineconeVectorStore as PineconeVectorStoreLlama
from langchain_pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_core.prompts import PromptTemplate
import tiktoken
import time
import os
from langchain_openai import ChatOpenAI


prompt = PromptTemplate(template="""You are Rachel, an expert paralegal tasked with helping on the case People of the State of Illinois v. Logan Emerson. This is a summary of the document you will answer questions about:
{context}
This is the statement of the case:
                        
On October 27, 2021, Logan Emerson, 18 years of age, texted Rory Hollis, 18 years of
age, to meet around 6:30 p.m. at the Lincolnville Community Theater for an impromptu
rehearsal for the upcoming local community theater production - How the Garden of the Gods
Were Won!. Both teenagers were actors in the play.
During the rehearsal, around 6:53 p.m., a janitor, Cameron Smith, heard a loud BOOM
coming from the stage. Thereafter, Logan Emerson called 911 and told police that he had
accidently shot Rory Hollis. Hollis was pronounced dead at the scene and Logan Emerson was
taken into custody by the Lincoln County Police Department where he gave a statement after
waiving his Miranda rights.
The State has charged Emerson with First Degree murder under 720 ILCS 5/9-1(a)(2)
and Involuntary Manslaughter under 720 ILCS 5/9-3(a). Emerson has pleaded not guilty to both
charges.

                        
Always start your answer with the phrase 'Answer:' . Do not include or reference quoted content verbatim in the answer. Don't say "According to Quote [1]" when answering. Instead make references to quotes relevant to each section of the answer solely by adding their bracketed numbers at the end of relevant sentences.


When writing motions, briefs or answering question that require sources, answer the question then find the quotes from the document that are most relevant to answering the question, and then print the quotes in numbered order.

If there are no relevant quotes, write "No relevant quotes" instead. 

Format your response in html. Use <p>, <b>, <li>, <ol> tags, etc. to format your response. Do not include any code or code snippets in your response. Be confident with your response and do not ask for clarification or confirmation. Do not include any personal opinions or beliefs in your response. Be concise and to the point. Do not include any irrelevant information.


Thus, the format of your overall response should look like what's shown between the <div></div> tags. Make sure to follow the formatting and spacing exactly. Include which files you are pulling information from.

Answer:
<div>
<p>
<br>
Company X earned $12 million. [1] Almost 90% of it was from widget sales. [2]    
</p>          

<br>
<ol>
<li>[1]  <b>(company_x_report.pdf)</b></li>
<li>[2]  <b>(company_gadgets_brief.pdf) </b></li>
</ol>

</div>


The quotes section should be the last part of your response.                        

Format your response in html. Use <p>, <b>, <li>, <ol> tags, etc. to format your response. Do not include any code or code snippets in your response. Be confident with your response and do not ask for clarification or confirmation. Do not include any personal opinions or beliefs in your response. Be concise and to the point. Do not include any irrelevant information.



This is the question: {question}
                        


     

                        """, input_variables=['question', 'context'])


def format_docs(docs):
    return_string = ''

    for doc in docs:
        return_string = return_string + \
            "Page Metadata: " + str(doc.metadata) + "\n"
        return_string = return_string + " " + doc.text + "\n\n"
    return return_string


def get_rachel_answer(question: str, pinecone_index: str):
    print("Start of function")
    pc = PineconeGRPC()
    start_time = time.time()

    print("Setting up Vector Store")
    vector_store = PineconeVectorStoreLlama(
        pinecone_index=pc.Index(pinecone_index))
    print("--Done in %s seconds" % (time.time() - start_time))

    print("Setting up Vector Index")
    start_time_vec_index = time.time()
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=OpenAIEmbedding(model_name='text-embedding-3-small'))
    print("--Done in %s seconds" % (time.time() - start_time_vec_index))

    print("Setting up Retriever")
    start_time_retriever = time.time()
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5,
                                     embed_model=OpenAIEmbedding(model_name='text-embedding-3-small'))
    print("--Done in %s seconds" % (time.time() - start_time_retriever))

    print("Getting Context")
    start_time_context = time.time()
    context = format_docs(retriever.retrieve(question))
    print("This is the context: \n" + context)
    print('--Done in %s seconds' % (time.time() - start_time_context))

    print("Setting up Rag Chain")
    start_time_rag_chain = time.time()

    llm = ChatAnthropic(model='claude-3-opus-20240229')

    rag_chain = prompt | llm | StrOutputParser()
    print('--Done in %s seconds' % (time.time() - start_time_rag_chain))

    prompt_str = prompt.to_json()['kwargs']['template']

    encoding = tiktoken.get_encoding("cl100k_base")

    prompt_len = len(encoding.encode(prompt_str))
    context_len = len(encoding.encode(context))
    question_len = len(encoding.encode(question))

    cost = (prompt_len + context_len + question_len)*0.000015

    print("input cost in dollars : " + str(cost))

    print("Getting Answer")
    start_time_answer = time.time()
    answer = ''
    for s in rag_chain.stream({'context': context, "question": question}):
        yield (s)
        answer += s
    print('--Done in %s seconds' % (time.time() - start_time_answer))

    print("End of function -- Total execution time: %s seconds" %
          (time.time() - start_time))

    answer_len = len(encoding.encode(answer))

    answer_cost = answer_len*0.000015

    print("answer cost: " + str(answer_cost))

    total_cost = answer_cost + cost

    print("total cost: " + str(total_cost))

    return answer


def similarity_search_for_documents(query: str, index_name: str, k: int):
    # Record start time
    start_time = time.time()

    # Initializing embeddings
    print("Initializing embeddings...")
    embeddings = OpenAIEmbeddings()
    print("--- %s seconds ---" % (time.time() - start_time))

    # Getting the vectorstore
    start_time = time.time()
    print("Getting the vector store from the existing index...")
    vectorStore = PineconeVectorStore.from_existing_index(
        embedding=embeddings, index_name=index_name)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Getting the similarity search documents
    start_time = time.time()
    print("Performing similarity search on the documents...")
    documents = vectorStore.similarity_search(query, k)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Similarity search for documents completed successfully.")
    return documents
