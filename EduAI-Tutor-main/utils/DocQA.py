from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from typing import TypedDict, List, Iterator, Dict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import FAISS


class GraphState(TypedDict):
    query: str
    messages: List[BaseMessage]
    response: str
    context: List[str]


class DocumentQA:

    def __init__(
        self,
        llm_model: str = "llama-3.3-70b-versatile",
        faiss_path: str = "faiss_index_local",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.llm = ChatGroq(model=llm_model, streaming=True)
        self.vectorstore = FAISS.load_local(
            faiss_path,
            embeddings=HuggingFaceEmbeddings(model_name=embed_model),
            allow_dangerous_deserialization=True,
        )
        self.graph = self.build_graph()

    def retriever_node(self, state: GraphState) -> GraphState:
        if not self.vectorstore:
            print("Warning: Vectorstore not initialized")
            state["context"] = []
            return state
        docs = self.vectorstore.similarity_search(state["query"], k=2)
        state["context"] = [doc.page_content for doc in docs]
        return state

    def memory_node(self, state: GraphState) -> GraphState:
        query = state["query"]
        state["messages"].append(HumanMessage(content=query))
        return state

    def llm_node(self, state: GraphState) -> GraphState:
        context = "\n\n".join(state["context"])
        messages = state["messages"] + [
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{state['query']}")
        ]
        response = self.llm.invoke(messages)
        state["messages"].append(AIMessage(content=response.content))
        state["response"] = response.content
        return state

    def build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("memory", self.memory_node)
        builder.add_node("retriever", self.retriever_node)
        builder.add_node("llm", self.llm_node)

        builder.set_entry_point("memory")
        builder.add_edge("memory", "retriever")
        builder.add_edge("retriever", "llm")
        builder.set_finish_point("llm")
        return builder.compile()

    def init_state(self) -> GraphState:
        return {
            "query": "",
            "messages": [
                SystemMessage(
                    content="You are a helpful and learned teacher. You help the users understand a study material. "
                    "Provide clear, concise, and accurate answers based on the context provided. "
                    "If the context doesn't contain the information needed, say so honestly."
                )
            ],
            "context": [],
            "response": "",
        }

    def restore_state(self, serialized_state: dict) -> GraphState:
        if serialized_state is None:
            return self.init_state()

        messages = []
        for msg in serialized_state.get("messages", []):
            if isinstance(msg, dict):
                msg_type = msg.get("type", "human")
                content = msg.get("content", "")
                if msg_type == "system":
                    messages.append(SystemMessage(content=content))
                elif msg_type == "ai":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
            else:
                messages.append(msg)

        return {
            "query": serialized_state.get("query", ""),
            "messages": messages,
            "context": serialized_state.get("context", []),
            "response": serialized_state.get("response", ""),
        }

    def serialize_state(self, state: GraphState) -> dict:
        serialized_messages = []
        for msg in state["messages"]:
            if isinstance(msg, SystemMessage):
                serialized_messages.append({"type": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                serialized_messages.append({"type": "ai", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                serialized_messages.append({"type": "human", "content": msg.content})

        return {
            "query": state.get("query", ""),
            "messages": serialized_messages,
            "context": state.get("context", []),
            "response": state.get("response", ""),
        }

    def run(self, query: str, state: GraphState) -> GraphState:
        state["query"] = query
        return self.graph.invoke(state)

    def run_stream(self, query: str, state: GraphState) -> Iterator[Dict]:
        state["query"] = query
        state = self.memory_node(state)
        state = self.retriever_node(state)

        context = "\n\n".join(state["context"])
        messages = state["messages"] + [
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{state['query']}")
        ]

        full_response = ""
        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                yield {"response_chunk": chunk.content}

        state["messages"].append(AIMessage(content=full_response))
        state["response"] = full_response
        yield {"state": state}
