from state import AgentState
from my_imports import *

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o")

def format_docs(docs):
    """Format retrieved documents into a single cohesive string"""
    return "\n\n".join(doc.page_content for doc in docs)


retrieved_vectors = PineconeVectorStore.from_existing_index(
index_name="mckinsey-report-1", embedding=embeddings
)
retriever = retrieved_vectors.as_retriever(search_kwargs={"k": 5})

prompt_template = ChatPromptTemplate.from_template("""
                    Answer the question based only on the following context.

                    {context}

                    Question: {question}

                    Provide a answer based on the context retrieved.
                    """)


def retrieval_chain(query: str):
    """
    Simple retrieval chain without LCEL.
    This is a manual method.
    """
    docs = retriever.invoke(query)

    context = format_docs(docs=docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)
    return response.content

def retrieve(state):
    docs = retriever.invoke(state['user_prompt'])
    state['retrieved_contexts'] = [doc.page_content for doc in docs]
    state['llm_response'] = retrieval_chain(query=state['user_prompt'])
    return state