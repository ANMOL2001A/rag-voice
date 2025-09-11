
from rag_system import retriever, chat_llm
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the following query based on the provided context.

Query: {query}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def main():
    """
    Continuously takes a user query from the terminal,
    searches the vector DB, and generates a response from the LLM.
    """
    print("Starting query terminal. Type 'exit' or 'quit' to end.")
    while True:
        try:
            query = input("\n>>> ")

            if query.lower().strip() in {"exit", "quit"}:
                print("Exiting...")
                break

            if not query:
                continue

            print("Searching for relevant documents...")
            similar_results = retriever.invoke(query)
            print(similar_results)
            
            if not similar_results:
                print("No relevant documents found.")
                continue

            context = "\n\n".join([doc.page_content for doc in similar_results])
            
            print("Top 3 results found. Generating response...")

            # 3. Pass results and query to the LLM
            formatted_prompt = prompt.format(query=query, context=context)
            
            # 4. Get response from LLM
            response = chat_llm.invoke(formatted_prompt)

            # 5. Print the response
            print("\n🤖 Assistant:\n" + "="*20)
            print(response.content)
            print("="*20)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

