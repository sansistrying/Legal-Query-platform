import os
from agents import Law_Agents
from tasks import Law_Tasks
from crewai import Crew, Process  # Assuming these are part of your project structure
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Define the main class for managing the law query tasks and agents
class Law_Crew:
    def __init__(self, query):
        self.query = query

    def run(self):
        # Initialize the agents and tasks
        agents = Law_Agents()
        tasks = Law_Tasks()

        # Create instances of different searcher agents
        pdf_searcher = agents.pdf_searcher()
        website_searcher = agents.website_searcher()
        web_searcher = agents.web_searcher()

        # Define tasks that use the searcher agents and the query
        website_search_task = tasks.website_search_task(website_searcher, self.query)
        web_search_task = tasks.web_search_task(web_searcher, self.query)
        pdf_search_task = tasks.pdf_search_task(pdf_searcher, self.query)

        # Initialize the Crew with agents and tasks
        crew = Crew(
            agents=[website_searcher, pdf_searcher, web_searcher],  # List of agents
            tasks=[website_search_task, pdf_search_task, web_search_task],  # List of tasks
            verbose=1,  # Verbosity level for logging
            full_output=False,  # Whether to return full output
            manager_llm=ChatOpenAI(  # Language model manager
                temperature=0,  # Model temperature setting
                model="gpt-3.5-turbo-0125",  # Model specification
                api_key=os.environ["OPENAI_API_KEY"],  # API key from environment variable
            ),
            process=Process.hierarchical,  # Process type for task management
            memory=True,  # Whether to use memory
        )

        # Start the process and pass the query as input
        result = crew.kickoff(inputs={"query": self.query})
        return result

if __name__ == "__main__":
    # Entry point for the script
    print("## Welcome to Law Query")
    print("-------------------------------")
    query = input("Enter your question: ")  # Get the user's query
    law_crew = Law_Crew(query)  # Instantiate the Law_Crew class with the query
    result = law_crew.run()  # Run the process and get the result
    print("\n\n########################")
    print(result)
