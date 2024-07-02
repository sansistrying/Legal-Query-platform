import os
from crewai import Task
from agents import Law_Agents
from tasks import Law_Tasks
from crewai import Crew, Process
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Setting Environment Variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["TAVILY_API_KEY"] = "your-api-key"

class Law_Crew:
    def __init__(self, query):
        self.query = query

    def run(self):
        agents = Law_Agents()
        tasks = Law_Tasks()

        # Create individual search agents
        pdf_searcher = agents.pdf_searcher()
        website_searcher = agents.website_searcher()
        web_searcher = agents.web_searcher()
        master_agent = agents.master_agent()

        # Create tasks for each search agent with the query
        website_search_task = tasks.website_search_task(website_searcher, self.query)
        web_search_task = tasks.web_search_task(web_searcher, self.query)
        pdf_search_task = tasks.pdf_search_task(pdf_searcher, self.query)

        # Define the master task that summarizes the content
        master_task = Task(
            description='summarize the content given to you and add the details such as article number or citation of a law and links etc in the answer from the content',
            expected_output='An answer to the question in 2 to 3 lines and be in detail.An answer along with list of urls and pdf from where the answer was taken from',
            agent=master_agent,
            context=[website_search_task, web_search_task, pdf_search_task]
        )

        crew = Crew(
            agents=[master_agent],
            tasks=[pdf_search_task, website_search_task, web_search_task, master_task],
            verbose=1,
            full_output=True,
            process=Process.sequential,
            output_log_file=True,
        )

        result = crew.kickoff(inputs={'query': self.query})
        return master_task.output.exported_output

if __name__ == "__main__":
    print("Welcome to Law Query Platform")
    print('-----------------------------')
    query=input("Enter your query:")
    law_crew=Law_Crew(query)
    result=law_crew.run()
    print(result)
