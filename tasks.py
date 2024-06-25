from crewai import Task
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.website_search_tool import website_search
from tools.pdf_search_tool import PDF_search
class Law_Tasks():
    def website_search_task(self,agent,query):
        return Task(
            description = 'Find answer to query if possible. If answer is I do not know stop and move on to next task. Do not hallucinate. Query is as follows, {query}',
            expected_output = 'An answer to the question which is taken from the data provided',
            agent = agent
        )
    def web_search_task(self,agent,query):
       
        return Task(
            description = 'Find answer to query asked. Query is as follows, {query}',
            expected_output = 'An answer to the question in 2 to 3 lines if found and a list of the urls from where the content was taken from. Do not reuse the same input, just give the answer directly',
            agent = agent,
               )
    
    def pdf_search_task(self,agent,query):      
        return Task(
            description = 'Find answer to query by using the tool and searching for answer in the given content. Query is as follows, {query}. Give the source of the answer as well',
            expected_output = 'An answer to the question in 2 to 3 lines.',
            agent = agent
        )
