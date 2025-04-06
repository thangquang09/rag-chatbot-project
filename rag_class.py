import logging
import os
from typing import Annotated, Literal, Sequence

from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from utils import get_n_user_queries
from vectorstore import VectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_times: int
    retrieval_query = str


class WorkFlow:
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        model_provider: str = "google_genai",
        # **kwargs,
    ):
        self.model = model
        self.model_provider = model_provider
        # self.kwargs = kwargs
        # self.vectorstore, self.retriever = get_vectorstore_retriever()

        #### NEW CODE
        self.vector_store = VectorStore(
            persist_directory="test_vectorstore_folder",
            collection_name="test_collection",
        )
        self.retriever = self.vector_store.retriever

        #### END OF NEW CODE
        self.retriever_tool = create_retriever_tool(
            retriever=self.retriever,
            name="retriever_tool",
            description="Retrieve relevant documents from the vectorstore",
        )
        self.tools = [self.retriever_tool]

    def reinitialize_vectorstore(self, collection_name: str):
        """Reinitialize the vectorstore with a new collection name."""
        self.vector_store.collection_name = collection_name
        self.vector_store.create_vectorstore(reload_vectordb=False)
        self.retriever = self.vector_store.retriever
        self.retriever_tool = create_retriever_tool(
            retriever=self.retriever,
            name="retriever_tool",
            description="Retrieve relevant documents from the vectorstore",
        )
        self.tools = [self.retriever_tool]

    def _get_llm(self, **kwargs):
        model_kwargs = {
            "model": self.model,
            "model_provider": self.model_provider,
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 0.95),
            "streaming": kwargs.get("streaming", False),
        }

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        model = init_chat_model(**model_kwargs)
        return model

    def grade_documents(self, state: State) -> Literal["generate", "rewrite"]:
        """Check if documents are relevant to query"""
        rewrite_times = state.get("rewrite_times", 0)
        if rewrite_times >= 2:
            logging.info(
                "##Grading Task: Rewrite times exceeded return 'generate' task"
            )
            return "generate"

        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

        messages = state["messages"]
        docs = messages[-1].content
        question = get_n_user_queries(messages, 1)

        if question == "":
            raise ValueError("question is None")
        logging.info(f"##Grading task: Question: {question}")
        logging.info(f"##Grading task: Docs: {docs[:30]}...")

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            Even if the document contains only PARTIALLY relevant information or background context
            that might help answer the question, consider it relevant.
            Give a binary score 'yes' or 'no' to indicate whether the document has ANY relevance to the question.""",
            input_variables=["context", "question"],
        )
        
        llm = self._get_llm(temperature=0.3).with_structured_output(
            Grade
        )
        chain = prompt | llm
        scored_result = chain.invoke(
            {"question": question, "context": docs}
        ).binary_score

        logging.info(f"##Grading Task: scored_result = {scored_result}")

        if scored_result == "yes":
            logging.info("##Grading Task: Docs relevant")
            return "generate"
        else:
            logging.info("##Grading Task: Docs not relevant")
            return "rewrite"

    def agent(self, state: State) -> State:
        logging.info("##Agent Task: Call")

        system_message = SystemMessage(
            content="""You are an intelligent assistant named ChatChatAI with access to a document retrieval tool to find information in a knowledge base.
            Please answer the user's questions, if you don't know the answer, please use the retrieval tool to find relevant documents in vectorstore.
            """
        )

        messages = [
            msg
            for msg in state["messages"]
            if isinstance(msg, (HumanMessage, AIMessage))
            and not (
                isinstance(msg, AIMessage)
                and msg.content == ""
                and msg.additional_kwargs.get("function_call")
            )
        ]

        messages.insert(0, system_message)
        llm = self._get_llm(temperature=0.0)
        llm_with_tool = llm.bind_tools(self.tools)
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}

    def rewrite(self, state: State) -> State:
        """Rewrite the user query for better documents retrieval"""
        logging.info("##Rewrite Task: Call")

        current_rewrites = state.get("rewrite_times", 0)
        new_rewrite_count = current_rewrites + 1

        messages = state["messages"]
        original_query = get_n_user_queries(messages, 1)
        logging.info(f"##Rewrite Task: Original query: {original_query}")
        logging.info(f"##Rewrite Task: Attempt {new_rewrite_count}")

        class RewrittenQuery(BaseModel):
            query: str = Field(description="The rewritten search query", min_length=3)
            reasoning: str = Field(
                description="Explanation of how the query was improved", default=""
            )

        prompt = PromptTemplate(
            template="""You are an AI assistant helping to improve search queries.
            Original query: {query}
            
            Rewrite this query to:
            1. Be more specific and detailed
            2. Include key terms that might appear in relevant documents
            3. Focus on the core information need
            
            Provide:
            1. The rewritten query
            2. A brief explanation of how you improved it
            """,
            input_variables=["query"],
        )

        llm = self._get_llm(temperature=0.0).with_structured_output(RewrittenQuery)
        chain = prompt | llm

        try:
            result = chain.invoke({"query": original_query})
            rewritten_query = result.query
            reasoning = result.reasoning
            logging.info(f"##Rewrite Task: Rewritten query: {rewritten_query}")
            logging.info(f"##Rewrite Task: Reasoning: {reasoning}")

            return {
                "messages": [HumanMessage(content=rewritten_query)],
                "rewrite_times": new_rewrite_count,
            }
        except Exception as e:
            logging.error(f"##Rewrite Task: Error: {e}")
            return {
                "messages": [HumanMessage(content="Can't rewrite the query")],
                "rewrite_times": new_rewrite_count,
            }

    def generate(self, state: State) -> State:
        """Generate a response based on the retrieved documents"""
        logging.info("##Generate Task: Call")
        messages = state["messages"]
        user_queries = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
        
        previous_question = " ".join(user_queries[:-1])
        previous_question = previous_question.replace("If you don't know the answer, please retrieve the documents from the vector store.", "")
        
        latest_question = user_queries[-1]
        logging.info(f"##Generate Task: Latest question: {latest_question}")

        docs = messages[-1].content

        try:
            llm = self._get_llm(temperature=0.0)
            prompt_template = """You are a helpful assistant answering the user's most recent question based on the provided context.

    Context information:
    {context}

    Previous question:
    {previous_question}

    FOCUS ON ANSWERING THIS SPECIFIC QUESTION: {question}

    Provide a comprehensive answer using only information from the context. If the context doesn't contain relevant information, say so clearly."""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "previous_question", "question"],
            )
            
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"context": docs, "previous_question": previous_question, "question": latest_question})

            if response:
                logging.info(f"##Generate Task: Response: {response[:100]}...")  # Log just beginning for large responses
            else:
                logging.info("##Generate Task: No response")
                response = "I couldn't generate a response based on the available information."
                
            return {"messages": [AIMessage(content=response)]}  # Wrap in AIMessage
        except Exception as e:
            logging.error(f"##Generate Task: Error generating response: {e}")
            return {"messages": [AIMessage(content="I encountered an error processing your request.")]}

    def get_workflow(self) -> StateGraph:
        workflow = StateGraph(State)

        workflow.add_node("agent", self.agent)
        retrieve = ToolNode([self.retriever_tool])
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("rewrite", self.rewrite)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "retrieve", END: END}
        )
        workflow.add_conditional_edges("retrieve", self.grade_documents)
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        graph = workflow.compile()
        return graph


if __name__ == "__main__":
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
    workflow = WorkFlow()
    graph = workflow.get_workflow()
    step1: State = graph.invoke(inputs)

    step1["messages"].append(HumanMessage(content="What is my name?"))

    step2: State = graph.invoke(step1)

    print(step2)
