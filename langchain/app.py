import json
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Define the schema for parsing output
class Review(BaseModel):
    avaliacao: Literal["positivo", "negativo"]
    tags: List[str]

def load_dataset(file_path: str):
    """Load the dataset from the given file path."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["positive"][:100] + data["negative"][:100]

def main():
    # Load dataset
    dataset = load_dataset("movie_review_dataset.json")

    # Initialize the OpenAI Chat model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Define the output parser using Pydantic
    parser = PydanticOutputParser(pydantic_object=Review)

    # Define the prompt template
    system_message_template = (
        "Você é um assistente de avaliação de filmes. "
        "Você irá receber uma avaliação de um filme e irá classificar "
        "se a avaliação é positiva ou negativa e gerar tags sobre a avaliação."
    )
    human_message_template = (
        "Avaliação: {review}\n\n"
        "Responda no seguinte formato JSON:\n{format_instructions}"
    )

    system_message = SystemMessagePromptTemplate.from_template(system_message_template)
    human_message = HumanMessagePromptTemplate.from_template(
        human_message_template, 
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    def process_review(review):
        messages = prompt.format_messages(review=review)
        response = llm(messages)
        parsed_response = parser.parse(response.content)
        return parsed_response.model_dump()

    results = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_review, dataset))

    # Save results to a file
    with open("results_chain.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
