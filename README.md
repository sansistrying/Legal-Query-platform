# Legal-Query-platform

## Project Overview

This project provides powerful tools designed to facilitate efficient document searching and querying from websites and PDF files. Leveraging advanced natural language processing techniques, these tools can extract relevant information, summarize content, and provide accurate answers to user queries. The project includes:

- **website_search_tools.py**: Tool for extracting and querying information from websites.
- **pdf_search_tool.py**: Tool for extracting and querying information from PDF documents.
- **law_tasks.py**: Defines tasks for website and PDF searches.
- **law_agents.py**: Defines agents specialized in legal document searches.
- **law_crew.py**: Integrates agents and tasks to handle user queries through a streamlined process.

## Tools Description

### 1. Website Search Tool
**File:** website_search_tools.py

The Website Search Tool scrapes website content, stores it, and makes it searchable through natural language queries. Key features include:

- **Sitemap Parsing**: Fetches and parses the sitemap XML of a website to extract URLs.
- **Document Loading**: Loads web documents from the extracted URLs.
- **Data Storage**: Stores website data in a CSV file for persistent storage and future use.
- **Document Processing**: Converts loaded documents into a format suitable for embedding and searching.
- **Text Splitting**: Splits documents into smaller chunks for efficient processing.
- **Embeddings and Vector Search**: Utilizes OpenAI embeddings and FAISS vector search to find relevant documents based on user queries.
- **Question Answering**: Implements a question-answering chain to provide precise answers from the relevant documents.

**Usage Example:**
```python
website_tool = website_search()
result = website_tool._run("What is the latest update on the website?")
print(result)
```

### 2. PDF Search Tool
**File:** pdf_search_tool.py

The PDF Search Tool searches and queries information from PDF documents. Key features include:

- **PDF Reading**: Reads and extracts text from PDF files in specified directories.
- **Text Extraction**: Extracts text from each page of the PDFs.
- **Text Splitting**: Splits extracted text into manageable chunks for efficient processing.
- **Embeddings and Vector Search**: Uses OpenAI embeddings and FAISS vector search to find relevant documents based on user queries.
- **Question Answering**: Implements a question-answering chain to provide precise answers from the relevant documents.

**Usage Example:**
```python
pdf_tool = PDF_search()
result = pdf_tool._run("Summarize the contract details from the PDFs.")
print(result)
```

### 3. Law Tasks
**File:** law_tasks.py

This file defines various tasks related to legal document searches. Key features include:

- **Website Search Task**: Creates tasks for searching information on websites.
- **Web Search Task**: Creates tasks for conducting general web searches.
- **PDF Search Task**: Creates tasks for searching information within PDF documents.

**Usage Example:**
```python
tasks = Law_Tasks()
task = tasks.website_search_task(agent, "Find latest legal updates.")
print(task.description)
```

### 4. Law Agents
**File:** law_agents.py

This file defines agents specialized in legal document searches. Key features include:

- **PDF Searcher**: An agent specialized in scrutinizing legal documents in PDF format.
- **Website Searcher**: An agent specialized in scouring legal websites and online databases for relevant information.
- **Web Searcher**: An agent specialized in conducting web searches to retrieve legal information.

**Usage Example:**
```python
agents = Law_Agents()
pdf_agent = agents.pdf_searcher()
print(pdf_agent.role)
```

### 5. Law Crew
**File:** law_crew.py

This file integrates agents and tasks to handle user queries through a streamlined process. Key features include:

- **Initialization**: Sets up agents and tasks based on user queries.
- **Query Processing**: Processes user queries using the integrated agents and tasks.

**Usage Example:**
```python
if __name__ == "__main__":
    st.title("Law Query")
    query = st.text_input("Question:")
    if st.button("Submit"):
        law_crew = Law_Crew(query)
        result = law_crew.run()
        st.write("Result:")
        st.write(result)
```

## Installation and Setup

**Clone the Repository:**
```bash
git clone https://github.com/your-repo/document-search-tools.git
cd document-search-tools
```


**Set Up API Keys:**
Ensure you have your OpenAI API key set up in your environment:
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

## Configuration

- **Website URLs**: Update the `url` variable in `website_search_tools.py` with the sitemap URL of the website you want to scrape.
- **PDF Directory**: Update the directory path in `pdf_search_tool.py` to point to the location of your PDF files.

## Running the Tools

- **Running the Script:**: The script will leverage the CrewAI framework to process the idea and generate a landing page.
```bash
streamlit run app.py
```

## Conclusion

These tools are designed to streamline the process of extracting, storing, and querying information from websites and PDF documents. By leveraging advanced NLP techniques, they ensure that users can efficiently find and retrieve relevant information with ease. This project aims to significantly enhance productivity and decision-making processes within the organization.

## License
This project is released under the MIT License.
