"""Combined toolkit module created for small helpers.

This file groups the router and two simple toolkits together and provides
an entrypoint main() similar to the original src/main.py.
"""
import sys

def route_query(query: str) -> str:
    """Route the query to the appropriate toolkit.

    For queries that include 'calendar', route to the calendar toolkit,
    otherwise use the rag toolkit.
    """
    if 'calendar' in query.lower():
        return 'calendar'
    return 'rag'


def calendar_process_query(query: str) -> str:
    return "Calendar toolkit handling query: " + query


def rag_process_query(query: str) -> str:
    return "RAG toolkit handling query: " + query


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ")

    toolkit = route_query(query)
    if toolkit == 'calendar':
        response = calendar_process_query(query)
    elif toolkit == 'rag':
        response = rag_process_query(query)
    else:
        response = "No toolkit available for this query."

    print(response)


if __name__ == '__main__':
    main()

