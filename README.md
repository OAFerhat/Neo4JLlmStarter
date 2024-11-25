# Neo4j LLM Integration Learning Project

This project is a learning exercise focused on integrating Large Language Models (LLMs) with Neo4j graph databases using LangChain. It follows the Neo4j GraphAcademy curriculum for combining the power of graph databases with modern AI capabilities.

## Project Overview

The project demonstrates how to:
- Connect to a Neo4j database using Python
- Integrate OpenAI's LLMs via LangChain
- Perform graph-based operations with natural language processing
- Build intelligent graph applications

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure

- `src/` - Source code directory
- `notebooks/` - Jupyter notebooks for experiments and learning
- `requirements.txt` - Project dependencies
- `.env` - Environment variables (not tracked in git)

## Getting Started

1. Set up your Neo4j database (local or AuraDB)
2. Configure your environment variables
3. Follow the examples in the notebooks directory

## Resources

- [Neo4j GraphAcademy](https://graphacademy.neo4j.com/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [Neo4j Python Driver Documentation](https://neo4j.com/docs/python-manual/current/)
