from crewai import Task
from langchain_community.tools.tavily_search import TavilySearchResults

class Law_Tasks():
    def website_search_task(self, agent, query):
        """
        Create a task for website search.

        Args:
            agent: The agent responsible for executing the task.
            query (str): The query to search for on the website.

        Returns:
            Task: A task object for website search.
        """
        return Task(
            description=f'Find an answer to the query asked by searching the data that we have stored and analyze. Your final answer should be a detailed and brief reply to the query. The query is as follows: "{query}"',
            expected_output='A well described and brief answer to the query asked.',
            agent=agent
        )

    def web_search_task(self, agent, query):
        """
        Create a task for web search.

        Args:
            agent: The agent responsible for executing the task.
            query (str): The query to search for on the web.

        Returns:
            Task: A task object for web search.
        """
        return Task(
            description=f'Find an answer to the query asked by searching the web. Your final answer should be a detailed and brief reply to the query. The query is as follows: "{query}"',
            expected_output='A well described and brief answer to the query asked.',
            agent=agent
        )

    def pdf_search_task(self, agent, query):
        """
        Create a task for PDF search.

        Args:
            agent: The agent responsible for executing the task.
            query (str): The query to search for in PDF documents.

        Returns:
            Task: A task object for PDF search.
        """
        return Task(
            description=f'Find an answer to the query asked by searching the PDFs. Your final answer should be a detailed and brief reply to the query. The query is as follows: "{query}"',
            expected_output='A well described and brief answer to the query asked.',
            agent=agent
        )
