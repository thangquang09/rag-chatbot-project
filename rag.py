from typing import Sequence, Annotated, Literal
from typing_extensions import TypedDict
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langgraph.graph import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from vectorstore import get_vectorstore_retriever
import os
import logging
import pprint

load_dotenv()
logging.basicConfig(level=logging.INFO)

# VECTOR DATABASE & RETRIEVER TOOL

vectorstore, retriever = get_vectorstore_retriever()

retriever_tool = create_retriever_tool(
    retriever=retriever, name="retriever_tool", description="Return relevant documents"
)
## DEFINE TOOL
tools = [retriever_tool]


# Utils
def get_history_from_messages(messages: Sequence[BaseMessage]) -> str:
    chat_history = ""
    for message in messages:
        if message.type in ("human", "ai"):
            chat_history += f"Role: {message.type} - Content: {message.content}\n\n"

    return chat_history


# GET LLMS


def get_llm(model: str = "gemini-2.0-flash", model_provider: str = "google_genai"):
    model = init_chat_model(
        model=model,
        model_provider=model_provider,
        GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
    )
    return model


# STATE DEFINE
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # memory:


# NOTES AND EDGES


def grade_documents(state: State) -> Literal["generate", "rewrite"]:
    """Check if documents are relevant to query"""

    # Data Model
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

    messages = state["messages"]
    # find documents (last message)
    docs = messages[-1].content

    # find the newest user query
    question = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            question = message.content
            break
    if question is None:
        raise ValueError("question is None")
    logging.info(f"##Grading task: Question: {question}")
    logging.info(f"##Grading task: Docs: {docs[:30]}...")

    # prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # chain
    llm = get_llm()
    llm = llm.with_structured_output(Grade)

    chain = prompt | llm

    scored_result = chain.invoke({"question": question, "context": docs}).binary_score

    logging.info(f"##Grading Task: scored_result = {scored_result}")

    if scored_result == "yes":
        logging.info("##Grading Task: Docs relevant")
        return "generate"
    else:
        logging.info("##Grading Task: Docs not relevant")
        return "rewrite"


def agent(state: State) -> State:
    logging.info("##Agent Task: Call")
    messages = state["messages"]

    llm = get_llm()
    llm_with_tool = llm.bind_tools(tools)

    response = llm_with_tool.invoke(messages)

    return {"messages": [response]}


def rewrite(state: State) -> State:
    return state


def generate(state: State) -> State:
    return state


inputs = State(
    {
        "messages": [
            HumanMessage(content="Hello, my name is Thang"),
            AIMessage(content="Hello, how can I have you?"),
            HumanMessage(
                content="What does Lilian Weng say about the types of agent memory?"
            ),
        ]
    }
)

# WORKFLOW
if __name__ == "__main__":
    workflow = StateGraph(State)

    ## Node
    workflow.add_node("agent", agent)
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite)

    ## Edges
    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent", tools_condition, {"tools": "retrieve", END: END}
    )

    workflow.add_conditional_edges("retrieve", grade_documents)

    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", END)

    graph = workflow.compile()

    output = graph.invoke(inputs)
    output["messages"].append(HumanMessage(content="What is my name?"))

    output = graph.invoke(output)

    print(output)
