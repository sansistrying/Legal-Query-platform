import os  # Importing necessary modules and libraries
import streamlit as st
from agents import Law_Agents
from tasks import Law_Tasks
from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Law_Crew:
    def __init__(self, query):
        self.query = query  # Initialize with user query

    def run(self):
        # Initialize agents and tasks for law-related queries
        agents = Law_Agents()
        tasks = Law_Tasks()
        pdf_searcher = agents.pdf_searcher()
        website_searcher = agents.website_searcher()
        web_searcher = agents.web_searcher()

        # Define tasks based on user query
        website_search_task = tasks.website_search_task(website_searcher, self.query)
        web_search_task = tasks.web_search_task(web_searcher, self.query)
        pdf_search_task = tasks.pdf_search_task(pdf_searcher, self.query)

        # Configure and manage the crew for handling queries
        crew = Crew(
            agents=[website_searcher, pdf_searcher, web_searcher],
            tasks=[website_search_task, pdf_search_task, web_search_task],
            verbose=1,
            full_output=False,
            manager_llm=ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo-0125",
                api_key=os.environ["OPENAI_API_KEY"],
            ),
            process=Process.hierarchical,
            memory=True,
        )

        # Process user query using the crew
        result = crew.kickoff(inputs={"query": self.query})
        return result

if __name__ == "__main__":
    # Set up Streamlit app for user interaction
    st.title("Legal Query Platform")
    st.write("Enter your question below:")
    query = st.text_input("Question:")
    if st.button("Submit"):
        # Create Law_Crew instance and run query
        trip_crew = Law_Crew(query)
        result = trip_crew.run()
        st.write("Result:")
        st.write(result)
