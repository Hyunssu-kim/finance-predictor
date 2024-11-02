from typing import List, Dict, Optional, Any, Union
import json

from langchain_upstage import ChatUpstage as Chat
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import Optional, Dict, Union, List, Any

solar_pro = Chat(model="solar-pro")


def extracted_claimed_facts(
    text: str, llm: Optional[Chat] = solar_pro
) -> List[Dict[str, Any]]:
    """
    Extract claimed facts from the given text, including entities and their relationships.

    Args:
        text (str): The input text to extract facts from.
        llm (Optional[Chat]): The language model to use for extraction, if needed.

    Returns:
        List[Dict[str, Any]]: A list of extracted facts, where each fact is represented as a dictionary.
    """

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert fact extractor. Your task is to analyze the given text and extract a list of claimed facts, focusing on entities and their relationships. Extract precise and specific relations without categorizing them into predefined types.",
            ),
            (
                "human",
                """Extract the claimed facts from the following text, providing a list of dictionaries. Each dictionary should represent a fact and include keys for 'entity', 'relation', and 'value'. Be specific and precise with the relations.

Examples:
Input: "Albert Einstein developed the theory of relativity in 1915."
Output: [
    {{"entity": "Albert Einstein", "relation": "developed", "value": "theory of relativity"}},
    {{"entity": "theory of relativity", "relation": "developed in", "value": "1915"}}
]

Input: "The Eiffel Tower, completed in 1889, stands at a height of 324 meters."
Output: [
    {{"entity": "Eiffel Tower", "relation": "completed in", "value": "1889"}},
    {{"entity": "Eiffel Tower", "relation": "height", "value": "324 meters"}}
]

Now, extract facts from the following text:
{input_text}""",
            ),
            (
                "human",
                "Respond with a JSON array of fact dictionaries only, without any additional text.",
            ),
        ]
    )

    # Create the output parser
    output_parser = JsonOutputParser()

    # Create the chain
    chain = prompt | llm | output_parser

    # Run the chain
    result = chain.invoke({"input_text": text})

    return result


def search_context(
    text: str,
    claimed_facts: List[Dict[str, Any]],
    search_tool: DuckDuckGoSearchResults = DuckDuckGoSearchResults(),
    llm: Optional[Chat] = solar_pro,
) -> str:
    """
    Search for relevant information using claimed facts.

    Args:
        text (str): The original input text.
        claimed_facts (List[Dict[str, Any]]): The list of extracted claimed facts.
        search_tool (Any): The search tool to use for finding information (e.g., DuckDuckGoSearchResults).
        llm (Optional[Chat]): The language model to use for processing, if needed.

    Returns:
        str: The relevant context information found from the search.
    """

    # Step 1: Generate search keywords
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at generating concise and relevant search keywords. Your task is to analyze the given text and extracted facts, then produce a list of 3-5 search keywords or short phrases that would be most effective for finding additional context and verification information.",
            ),
            (
                "human",
                """Given the following text and extracted facts, generate a list of 3-5 search keywords or short phrases:

Text: {text}

Extracted Facts:
{facts}

Provide only the keywords or short phrases, separated by commas.""",
            ),
        ]
    )

    facts_str = "\n".join(
        [
            f"- {fact['entity']} {fact['relation']} {fact['value']}"
            for fact in claimed_facts
        ]
    )
    keywords_response = llm.invoke(prompt.format(text=text, facts=facts_str))

    # Parse the keywords from the response
    keywords = [kw.strip() for kw in keywords_response.content.split(",") if kw.strip()]

    # Step 2: Perform search using the generated keywords
    search_query = " ".join(keywords)
    search_results = search_tool.run(search_query)

    # Step 3: Return the search results
    return search_results


