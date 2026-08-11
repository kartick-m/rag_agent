import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from langchain_tavily import TavilySearch, TavilyCrawl, TavilyMap, TavilyExtract
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This is for react-agent-pydantic.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field
