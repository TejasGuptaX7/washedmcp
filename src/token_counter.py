"""
Token Counter Module

Counts tokens for measuring savings from semantic search.
Uses tiktoken for accurate OpenAI/Claude token counting,
falls back to character-based estimation if not available.
"""

from typing import Union

# Try to use tiktoken for accurate counting, fall back to estimation
try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4, Claude
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    _encoder = None


def count_tokens(text: str) -> int:
    """
    Count tokens in a string.
    
    Uses tiktoken if available (accurate), otherwise estimates
    based on ~4 characters per token (rough approximation).
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Number of tokens (int)
    """
    if not text:
        return 0
    
    if HAS_TIKTOKEN and _encoder:
        return len(_encoder.encode(text))
    else:
        # Rough estimation: ~4 chars per token for code
        return len(text) // 4


def count_tokens_batch(texts: list[str]) -> int:
    """
    Count total tokens across multiple strings.
    
    Args:
        texts: List of strings to count
        
    Returns:
        Total token count
    """
    return sum(count_tokens(text) for text in texts)


def calculate_savings(
    search_result_tokens: int,
    full_file_tokens: int
) -> dict:
    """
    Calculate token savings from using search vs reading full files.
    
    Args:
        search_result_tokens: Tokens in the search results
        full_file_tokens: Tokens if reading entire files
        
    Returns:
        Dict with tokens_used, tokens_saved, percent_saved
    """
    tokens_saved = max(0, full_file_tokens - search_result_tokens)
    
    if full_file_tokens > 0:
        percent_saved = round((tokens_saved / full_file_tokens) * 100, 1)
    else:
        percent_saved = 0.0
    
    return {
        "tokens_used": search_result_tokens,
        "tokens_without_search": full_file_tokens,
        "tokens_saved": tokens_saved,
        "percent_saved": percent_saved
    }


def estimate_file_tokens(file_path: str) -> int:
    """
    Estimate tokens in a file by reading it.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Estimated token count, 0 if file can't be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return count_tokens(content)
    except (IOError, OSError, UnicodeDecodeError):
        return 0


def format_token_stats(stats: dict) -> str:
    """
    Format token statistics for display.
    
    Args:
        stats: Dict from calculate_savings()
        
    Returns:
        Formatted string
    """
    return (
        f"📊 Token Stats:\n"
        f"   Used: {stats['tokens_used']:,} tokens\n"
        f"   Without search: {stats['tokens_without_search']:,} tokens\n"
        f"   Saved: {stats['tokens_saved']:,} tokens ({stats['percent_saved']}%)"
    )


if __name__ == "__main__":
    # Test the token counter
    test_code = '''
def check_reverse(s: str) -> bool:
    """Check if string is palindrome"""
    return s == s[::-1]
'''
    
    print(f"Using tiktoken: {HAS_TIKTOKEN}")
    print(f"Test code tokens: {count_tokens(test_code)}")
    
    # Test savings calculation
    stats = calculate_savings(
        search_result_tokens=150,
        full_file_tokens=2000
    )
    print(f"\n{format_token_stats(stats)}")
