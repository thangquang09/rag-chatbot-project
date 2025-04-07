from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage


def get_history_from_messages(messages: Sequence[BaseMessage]) -> str:
    """Get the chat history from messages"""
    if not messages:
        return ""
    chat_history = ""
    for message in messages:
        if message.type in ("human", "ai"):
            chat_history += f"Role: {message.type} - Content: {message.content}\n\n"

    ### Summarize if needed

    return chat_history


def get_n_user_queries(messages: Sequence[BaseMessage], n: int = 1) -> str:
    """Get the n most recent user queries from messages"""
    question = ""
    number_of_question = 0
    for idx, message in enumerate(reversed(messages)):
        if isinstance(message, HumanMessage):
            if number_of_question == 0:
                question = f"Newest user query: {message.content}"
            else:
                question += f"\n{message.content}"
            number_of_question += 1
            if number_of_question >= n:
                break
    return question

def estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """Count the number of tokens in messages"""
    total_content = "".join(msg.content for msg in messages)
    
    # Assuming 4 bytes per token (this is a rough estimate)
    estimated_tokens = len(total_content) // 4
    
    return estimated_tokens