import os
import httpx
import re
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pathlib import Path

# Step 1: Load GROQ API key from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Step 2: Read the query from input.txt
input_path = Path("input.txt")
try:
    with input_path.open("r", encoding="utf-8") as f:
        user_query = f.read().strip()
    if not user_query:
        raise ValueError("input.txt is empty")
except FileNotFoundError:
    raise FileNotFoundError(f"Input file '{input_path}' not found")
except UnicodeDecodeError:
    raise ValueError("Failed to decode input.txt. Ensure it is UTF-8 encoded.")

# Step 3: Define a simplified function to extract filename and content
def extract_filename_and_content(content):
    """
    Parses LLM output to extract a filename and clean content.
    It first tries to find a markdown-formatted filename (**file.ext**).
    If not found, it inspects the content to suggest a default filename.
    """
    content = content.strip()
    # Regex to find an optional filename and the content within code fences
    pattern = r"^(?:\*\*(.+?\.\w+)\*\*\n+)?(?:```(?:\w+)?\n|\'\'\'(?:\w+)?\n)?(.*?)(\n(?:```|\'\'\'))?$"
    match = re.match(pattern, content, re.DOTALL)
    
    filename = None
    parsed_content = content

    if match:
        # Extract the filename and the inner content if the pattern matches
        captured_filename, inner_content, _ = match.groups()
        if captured_filename:
            filename = captured_filename
        if inner_content:
            parsed_content = inner_content.strip()
    
    # If a filename was explicitly provided in the output, use it
    if filename:
        return parsed_content, filename
        
    # Default to a plain text file if no other type is detected
    return parsed_content, "output.txt"

# Step 4: Define the prompt template
prompt_template = """
You are an expert AI coding and text generation assistant. Process the following user query and generate the requested output in the appropriate format. Follow these instructions:

- For programming languages (Python, C, C++, Java, JavaScript), return only the valid, executable code without markdown code fences (``` or ''') or explanations.
- For JSON, return valid JSON without markdown code fences or explanations.
- For plain text, return a concise response as plain text without markdown.
- If a specific filename is relevant (e.g., for code or JSON), suggest it at the start of the output in the format **filename.ext** (e.g., **main.py** for Python, **data.json** for JSON).
- Ensure the output is clean and matches the requested format exactly, ready to be saved to a file.

Query: {user_query}
"""
prompt = PromptTemplate(
    input_variables=["user_query"],
    template=prompt_template
)

# Step 5: Set up the Groq client using LLaMA 3
try:
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        http_client=httpx.Client(verify=False)  # Disable SSL for local testing
    )
except Exception as e:
    raise Exception(f"Failed to initialize Groq client: {str(e)}")

# Step 6: Build the LangChain pipeline
chain = prompt | groq_llm

# Step 7: Send the query to the LLM and get the response
try:
    result = chain.invoke({"user_query": user_query}).content
except Exception as e:
    raise Exception(f"LLM processing failed: {str(e)}")

# Step 8: Parse the result to get the content and filename
try:
    # The function now returns only the two variables we need
    parsed_content, output_filename = extract_filename_and_content(result)
except Exception as e:
    raise Exception(f"Failed to parse LLM output: {str(e)}")

# Step 9: Save the result to the specified output file
output_path = Path(output_filename)
try:
    with output_path.open("w", encoding="utf-8") as f:
        f.write(parsed_content)
    print(f"✅ Output written to {output_path}")
except Exception as e:
    raise Exception(f"Failed to write to {output_path}: {str(e)}")
