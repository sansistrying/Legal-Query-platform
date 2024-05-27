from crewai import Agent
from tools.pdf_search_tool import PDF_search
from tools.website_search_tool import website_search
from langchain_community.tools.tavily_search import TavilySearchResults

class Law_Agents:
    # Define a class to manage law-related agents
    def pdf_searcher(self):
        """
        Create an agent for PDF search.

        Returns:
            Agent: An agent specialized in scrutinizing legal documents in PDF format.
        """
        return Agent(
            role="Legal Document Analyst",
            goal="To meticulously scrutinize legal documents in PDF format, extracting information to respond to queries and then return the answer to the question.",
            backstory="PDF_searcher is a seasoned legal professional with a specialization in document analysis. With a background in law and years of experience in handling intricate legal documents, PDF_searcher possesses a deep understanding of legal terminology and nuances. Their expertise allows them to navigate through complex texts efficiently, extracting key details and providing answers to complex legal inquiries.",
            tools=[PDF_search()],
            verbose=True,
        )

    def website_searcher(self):
        """
        Create an agent for website search.

        Returns:
            Agent: An agent specialized in scouring legal websites and online databases for relevant information.
        """
        return Agent(
            role="Legal Website Analyst",
            goal='To scour legal websites and online databases for relevant information, ensuring accurate and reliable responses to queries. If answer is not found, just say "I don\'t know".',
            backstory="website_searcher is a proficient legal researcher adept at navigating the vast landscape of online legal resources. With a background in law and extensive experience in online research, website_searcher possesses a keen eye for identifying credible sources and extracting pertinent information. Their familiarity with various legal platforms enables them to swiftly locate relevant data and deliver well-researched answers to complex legal questions.",
            tools=[website_search()],
            verbose=True,
        )

    def web_searcher(self):
        """
        Create an agent for web search.

        Returns:
            Agent: An agent specialized in conducting web searches to retrieve legal information.
        """
        return Agent(
            role="Legal Web Searcher",
            goal="To conduct web searches to retrieve legal information, providing responses to inquiries. try to provide as accurate as possible.",
            backstory="web_searcher is an adept legal researcher skilled in harnessing the power of internet search engines to gather comprehensive legal information. With a background in law and a passion for staying abreast of legal developments, web_searcher possesses a knack for conducting targeted searches and filtering through vast amounts of data to find relevant insights. Their proficiency in online research methodologies makes them a valuable asset in delivering timely and informed responses to legal queries.",
            tools=[TavilySearchResults(k=3)],
            verbose=True,
        )
