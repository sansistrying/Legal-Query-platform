from crewai import Agent
from tools.pdf_search_tool import PDF_search
from tools.website_search_tool import website_search
from langchain_community.tools.tavily_search import TavilySearchResults
import os

class Law_Agents():
    def pdf_searcher(self):
        return Agent(
            role='Legal Document Analyst',
            goal='To meticulously scrutinize legal documents in PDF format, extracting information to respond to queries and then return the answer to the question as well as the source of the answer.',
            backstory='PDF_searcher is a seasoned legal professional with a specialization in document analysis. With a background in law and years of experience in handling intricate legal documents, PDF_searcher possesses a deep understanding of legal terminology and nuances. Their expertise allows them to navigate through complex texts efficiently, extracting key details and providing answers to complex legal inquiries.',
            tools=[PDF_search()]
            )
    
    def website_searcher(self):
        return Agent(
            role='Legal Website Analyst',
            goal='To scour legal websites and online databases for relevant information, ensuring accurate and reliable responses to queries.',
            backstory='website_searcher is a proficient legal researcher adept at navigating the vast landscape of online legal resources. With a background in law and extensive experience in online research, website_searcher possesses a keen eye for identifying credible sources and extracting pertinent information. Their familiarity with various legal platforms enables them to swiftly locate relevant data and deliver well-researched answers to complex legal questions.',
            tools=[website_search()]
            )
    
    def web_searcher(self):
        return Agent(
            role='Legal Web Searcher',
            goal='To conduct web searches to retrieve legal information, providing responses to inquiries. try to provide as accurate as possible.',
            backstory='web_searcher is an adept legal researcher skilled in harnessing the power of internet search engines to gather comprehensive legal information. With a background in law and a passion for staying abreast of legal developments, web_searcher possesses a knack for conducting targeted searches and filtering through vast amounts of data to find relevant insights. Their proficiency in online research methodologies makes them a valuable asset in delivering timely and informed responses to legal queries.',
            tools=[TavilySearchResults(k=3)]
            )
    def master_agent(self):
        return Agent(
            role='Master Agent',
            goal='Check the output of the other tasks and prioritize which answer is the best suitable for the query asked by the user, adding details and citations as needed.',
            backstory='master_agent is a proficient legal detailed context summarizer with the ability to synthesize information from various sources to deliver the most accurate and comprehensive response.',
            
        )
 
