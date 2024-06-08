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
        self.query = query  # Initialize the class with the query

    def run(self):
        agents = Law_Agents()  # Instantiate the Law_Agents class
        tasks = Law_Tasks()  # Instantiate the Law_Tasks class

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
            description='summarize the content given to you and add the details such as article number or citation of a law etc in the answer from the content',
            expected_output='An answer to the question in 2 to 3 lines and be in detail.',
            agent=master_agent,
            context=[website_search_task, web_search_task, pdf_search_task]
        )

        # Create a crew with the defined agents and tasks
        crew = Crew(
            agents=[master_agent],
            tasks=[website_search_task, pdf_search_task, web_search_task, master_task],
            verbose=1,  # Enable verbose output for debugging
            full_output=True,  # Enable full output logging
            process=Process.sequential,  # Run tasks sequentially
            output_log_file=True,  # Enable output logging to a file
        )

        # Run the crew with the input query and return the result
        result = crew.kickoff(inputs={'query': self.query})

        return master_task.output.exported_output

if __name__ == "__main__":
    # Entry point for the script
    print("Welcome to Law Query Platform")
    print('-----------------------------')
    query = input("enter your query:")  # Prompt user for a query
    law_crew = Law_Crew(query)  # Instantiate the Law_Crew class with the query
    result = law_crew.run()  # Run the Law_Crew and get the result
    print(result)  # Print the result
