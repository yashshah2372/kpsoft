import streamlit as st
import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import logging
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotError(Exception):
    """Custom exception class for chatbot-specific errors"""
    pass

def load_environment_variables() -> bool:
    """Load and validate environment variables"""
    try:
        load_dotenv()
        if not os.getenv('OPENAI_API_KEY'):
            raise ChatbotError("Missing required environment variable: OPENAI_API_KEY")
        return True
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")
        st.error("Configuration error. Please contact administrator.")
        return False

def initialize_faiss() -> Optional[FAISS]:
    """Initialize FAISS vector store"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Error initializing FAISS: {str(e)}")
        st.error("Error initializing search system. Please contact administrator.")
        return None

def get_conversation_history(messages: List[Dict], limit: int = 5) -> List[Dict]:
    """Safely retrieve conversation history"""
    try:
        return messages[-limit:]
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        return []

def process_query(faiss_db: FAISS, query: str) -> Optional[str]:
    """Process query and retrieve context"""
    try:
        results = faiss_db.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in results])
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error("Error processing your query. Please try again.")
        return None

def main():
    """Main application with error handling"""
    try:
        # Load environment variables
        if not load_environment_variables():
            return

        # Initialize OpenAI API
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # Initialize FAISS
        faiss_db = initialize_faiss()
        if faiss_db is None:
            return

        # Streamlit UI setup
        st.set_page_config(page_title="ERP Help Chatbot")
        st.title("🤖 ERP Help Chatbot")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        query = st.chat_input("Ask me anything about ERP...")

        if query:
            # Process user input
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            # Get context
            context = process_query(faiss_db, query)
            if context is None:
                return

            # Prepare messages
            conversation_history = get_conversation_history(st.session_state.messages)
            messages = [
                {"role": "system", "content": """You are an ERP support chatbot. IMPORTANT INSTRUCTIONS:

1. Greeting Behavior:
   - If the user's message contains a greeting (hi, hello, hey, good morning, etc.), always start your response with an appropriate greeting
   - If the message has both a greeting and a question, respond with a greeting first, then answer their question
   - For standalone greetings, respond with a friendly welcome message and ask how you can help

2. Conversation Guidelines:
   - Maintain context from the previous messages in the conversation
   - Reference previous questions or answers when relevant
   - Use a conversational tone while maintaining professionalism
   - If referring to something mentioned earlier, be specific about what was discussed

3. Response Guidelines:
   - ONLY use the information provided in the context to answer questions
   - If the context doesn't contain the information needed, respond with: "I apologize, but I don't have enough information in my knowledge base to answer this question accurately. Please rephrase or ask another question."
   - DO NOT use any external knowledge or make assumptions beyond what's in the context
   - Keep responses clear, concise, and directly related to the ERP system information provided
   - When quoting from the context, use exact phrases to maintain accuracy
   - If the context contains multiple relevant pieces of information, synthesize them into a coherent response
   - Do not make up or infer information that isn't explicitly stated in the context
   - If you encounter any errors or issues, provide a clear and helpful error message to the user"""}
            ]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": f"Query: {query}\n\nRelevant Context: {context}"})

            try:
                # Get and display response
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    full_response = ""
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=800,
                        stream=True
                    )
                    
                    for chunk in response:
                        if "choices" in chunk and chunk["choices"]:
                            token = chunk["choices"][0]["delta"].get("content", "")
                            full_response += token
                            response_container.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                logger.error(f"Error in response generation: {str(e)}")
                st.error("Error generating response. Please try again.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please contact administrator.")

if __name__ == "__main__":
    main()