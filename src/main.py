import sys

from toolkit_router import route_query
import calendar_toolkit
import rag_toolkit

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ")

    toolkit = route_query(query)
    if toolkit == 'calendar_toolkit':
        response = calendar_toolkit.process_query(query)
    elif toolkit == 'rag_toolkit':
        response = rag_toolkit.process_query(query)
    else:
        response = "No toolkit available for this query."

    print(response)

if __name__ == '__main__':
    main()
