# agent/actions.py

from query.sql_pipeline import run_query_pipeline
from rag.narrative_retriever import get_relevant_narrative_chunks

def run_stats_tool(question: str) -> dict:
    """
    Executes the SQL pipeline to get structured stats.
    Returns: dict with "result", "sql", etc.
    """
    return run_query_pipeline(question)


def run_narrative_tool(question: str, top_k: int = 5) -> list[dict]:
    """
    Executes the narrative retriever.
    Returns: list of chunks with 'doc_title' and 'text'.
    """
    return get_relevant_narrative_chunks(question, top_k=top_k)