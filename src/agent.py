"""Agentic RAG with LangGraph (the Project #2 agent variant).

Instead of a fixed retrieve->generate chain, this is a StateGraph where the LLM
decides WHEN to search the document (the retriever is exposed as a tool), can
search multiple times, and keeps memory across turns via a checkpointer.

Graph:
    START -> agent --(tool call?)--> tools -> agent -> ... -> END
    (tools_condition routes to `tools` if the LLM emitted a tool call, else END)

Run:  python src/agent.py    (needs the FAISS index built + Ollama running)

Version/notes:
  - pip install langgraph
  - Tool-calling needs a function-calling-capable model. gemma4:e4b can be
    unreliable at emitting well-formed tool calls; if the agent won't call the
    tool, switch config.LLM_MODEL to a bigger one (e.g. gemma4:12b).
  - LangGraph import paths shift between versions - verify against your install.
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

import config
from rag import load_retriever, _format_docs

_retriever = load_retriever()


@tool
def search_docs(query: str) -> str:
    """Search the indexed document for passages relevant to the query.
    Use this to ground every factual answer; call it again with a refined query
    if the first results are insufficient."""
    return _format_docs(_retriever.invoke(query))


SYSTEM = SystemMessage(content=(
    "You are a document Q&A assistant. Use the search_docs tool to find relevant "
    "passages, then answer ONLY from what it returns. If the document doesn't "
    "contain the answer, say you don't know. Be concise."
))

llm = ChatOllama(model=config.LLM_MODEL, temperature=0).bind_tools([search_docs])


def agent_node(state: MessagesState):
    # prepend the system message once, at the front of the conversation
    msgs = state["messages"]
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SYSTEM] + msgs
    return {"messages": [llm.invoke(msgs)]}


def build_agent():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode([search_docs]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=MemorySaver())        # checkpointer = per-thread memory


def main():
    agent = build_agent()
    thread = {"configurable": {"thread_id": "cli-session"}}  # memory scope
    print("Agentic RAG (Ctrl-C to quit). Ask about the indexed document.")
    while True:
        try:
            q = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        result = agent.invoke({"messages": [HumanMessage(content=q)]}, thread)
        print(f"\nbot > {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
