import os
import streamlit as st
from crewai import Task
from agents import Law_Agents
from tasks import Law_Tasks
from crewai import Crew, Process
from dotenv import load_dotenv
from streamlit_chat import message

load_dotenv()

# Groq API base_url 
os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1' 

# Model you wish to use, see https://console.groq.com/docs/models 
os.environ["OPENAI_MODEL_NAME"] = 'llama3-70b-8192' 

# Your Groq API key
os.environ["OPENAI_API_KEY"] = 'api-key'  

os.environ["GROQ_API_KEY"] = 'api-key'  

os.environ["TAVILY_API_KEY"] = 'api-key'  
os.environ["AI21_API_KEY"] = 'api-key'  



class Law_Crew:
    def __init__(self, query):
        self.query = query  # Initialize the class with the query

    def run(self):
        # Instantiate the Law_Agents and Law_Tasks classes
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
            description='summarize the content given to you and add the details in the answer from the content',
            expected_output='An answer to the question in 2 to 3 lines and be in detail.',
            agent=master_agent,
            context=[pdf_search_task]
        )

        # Create a crew with the defined agents and tasks
        crew = Crew(
            agents=[master_agent],
            tasks=[pdf_search_task,master_task],
            verbose=1,  # Enable verbose output for debugging
            full_output=True,  # Enable full output logging
            process=Process.sequential,  # Run tasks sequentially
            output_log_file=True,  # Enable output logging to a file
        )

        # Run the crew with the input query and return the result
        result = crew.kickoff(inputs={'query': self.query})
        
        return master_task.output.exported_output  # Return the summarized result

#Streamlit app setup
st.title("Law Query Chatbot")  # Set the title of the Streamlit app
st.write('Ask law-related question and get a detailed answer!')  # Description of the app

# Initialize the session state for storing messages if not already initialized
if 'messages' not in st.session_state:
    st.session_state.messages = []
# Get user query input
user_query = st.text_input("You:")

if user_query:
    # Append user query to messages in session state
    st.session_state.messages.append({"message": user_query, "is_user": True})
    
    # Process the query with Law_Crew
    law_crew = Law_Crew(user_query)
    result = law_crew.run()
    
    # Append the result to messages in session state
    st.session_state.messages.append({"message": result, "is_user": False})

# Display messages in the chat interface
for msg in st.session_state.messages:
    if msg["is_user"]:
        message(msg["message"], is_user=True)  # Display user messages
    else:
        message(msg["message"], is_user=False)  # Display bot responses
