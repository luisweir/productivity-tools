def route_query(query: str) -> str:
    """Route the query to the appropriate toolkit.
    For queries that include 'calendar', route to the calendar toolkit, otherwise use rag_toolkit.
    """
    if 'calendar' in query.lower():
        return 'calendar_toolkit'
    return 'rag_toolkit'

if __name__ == '__main__':
    queries = [
        'what is my calendar events',
        'what AI tools can I use in Oracle?'
    ]
    for q in queries:
        print(q, '->', route_query(q))
