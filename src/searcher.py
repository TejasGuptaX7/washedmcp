"""Semantic code search using embeddings and vector database."""

import os
from src.embedder import embed_query
from src.database import init_db, search as db_search, get_stats
from src.token_counter import count_tokens, estimate_file_tokens, calculate_savings


def search_code(
    query: str,
    persist_path: str = "./.washedmcp/chroma",
    top_k: int = 5
) -> list[dict]:
    """
    Search for code snippets semantically similar to the query.

    Args:
        query: Natural language search query
        persist_path: Path to the ChromaDB persistence directory
        top_k: Number of results to return

    Returns:
        List of matching code snippets with metadata and similarity scores
    """
    try:
        # Initialize database
        init_db(persist_path=persist_path)

        # Embed the query
        query_embedding = embed_query(query)

        # Search the database (returns formatted results)
        results = db_search(query_embedding, top_k=top_k)

        return results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def search_code_with_stats(
    query: str,
    persist_path: str = "./.washedmcp/chroma",
    top_k: int = 5
) -> tuple[list[dict], dict]:
    """
    Search for code and calculate token savings.

    Args:
        query: Natural language search query
        persist_path: Path to the ChromaDB persistence directory
        top_k: Number of results to return

    Returns:
        Tuple of (results, token_stats)
        token_stats contains: tokens_used, tokens_without_search, tokens_saved, percent_saved
    """
    results = search_code(query, persist_path, top_k)
    
    if not results:
        return results, {
            "tokens_used": 0,
            "tokens_without_search": 0,
            "tokens_saved": 0,
            "percent_saved": 0.0
        }
    
    # Calculate tokens in search results (the code snippets returned)
    result_code = "\n".join(r.get("code", "") for r in results)
    tokens_used = count_tokens(result_code)
    
    # Estimate tokens if we had to read the full files
    # Get unique file paths from results
    file_paths = list(set(r.get("file_path", "") for r in results if r.get("file_path")))
    tokens_without_search = sum(estimate_file_tokens(fp) for fp in file_paths)
    
    # If we couldn't read files, estimate based on typical file size
    if tokens_without_search == 0:
        # Assume average file is ~500 tokens, and we'd need to read ~3 files to find what we want
        tokens_without_search = 500 * max(3, len(file_paths))
    
    # Calculate savings
    token_stats = calculate_savings(tokens_used, tokens_without_search)
    
    return results, token_stats


def is_indexed(persist_path: str = "./.washedmcp/chroma") -> bool:
    """
    Check if the database exists and contains indexed items.

    Args:
        persist_path: Path to the ChromaDB persistence directory

    Returns:
        True if database exists and has items, False otherwise
    """
    try:
        if not os.path.exists(persist_path):
            return False

        init_db(persist_path=persist_path)
        stats = get_stats()
        return stats["total_functions"] > 0

    except Exception:
        return False


if __name__ == "__main__":
    if is_indexed():
        print("Database is indexed. Searching for 'palindrome'...\n")
        results = search_code("palindrome")

        if results:
            print(f"Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"--- Result {i} ---")
                print(f"Function: {result['function_name']}")
                print(f"File: {result['file_path']}")
                print(f"Lines: {result['line_start']}-{result['line_end']}")
                print(f"Summary: {result['summary']}")
                print(f"Similarity: {result['similarity']}")
                print(f"Code:\n{result['code']}")
                print()
        else:
            print("No results found.")
    else:
        print("Not indexed yet")
