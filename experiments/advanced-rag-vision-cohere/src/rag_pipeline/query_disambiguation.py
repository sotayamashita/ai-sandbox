import os

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def assess_ambiguity_thought(query: str) -> str:
    """
    Thought: Assess whether the query is ambiguous.
    """
    print("[Thought] Assessing if the query is ambiguous.")
    return f"""Determine if the following question is ambiguous. Respond with "Yes" if it is ambiguous, otherwise respond with "No". Question: "{query}" """


def assess_ambiguity_action(prompt: str) -> str:
    """
    Action: Use LLM to assess ambiguity.
    """
    print("[Action] Evaluating ambiguity using LLM.")
    response = gemini_client.models.generate_content(
        model="gemini-1.5-pro", contents=[prompt]
    )
    return response.text.strip().lower()


def generate_clarifying_question_thought(query: str) -> str:
    """
    Thought: Decide to generate a clarifying question.
    """
    print("[Thought] Generating a clarifying question for the ambiguous query.")
    return f"""
        The following question is ambiguous. Generate one clarifying question to better understand the user's intent.
        Please respond in the same language as the user's <QUESTION>.
        <QUESTION>
        {query}
        </QUESTION>
    """


def generate_clarifying_question_action(prompt: str) -> str:
    """
    Action: Use LLM to generate a clarifying question.
    """
    print("[Action] Generating clarifying question using LLM.")
    response = gemini_client.models.generate_content(
        model="gemini-1.5-pro", contents=[prompt]
    )
    return response.text.strip()


def formulate_final_query_thought(
    original_query: str, clarifying_question: str, user_response: str
) -> str:
    """
    Thought: Formulate a clear query based on user's response.
    """
    print("[Thought] Formulating a clear query based on the user's response.")
    return f"""
        <ORIGINAL_QUESTION>
        {original_query}
        </ORIGINAL_QUESTION>
        <CLARIFYING_QUESTION>
        {clarifying_question}
        </CLARIFYING_QUESTION>
        <USER_RESPONSE>
        {user_response}
        </USER_RESPONSE>
        Based on the above, generate a clear and specific question.
        Please respond in the same language as the user's <ORIGINAL_QUESTION>.
    """


def formulate_final_query_action(prompt: str) -> str:
    """
    Action: Use LLM to generate the final disambiguated query.
    """
    print("[Action] Generating final disambiguated query using LLM.")
    response = gemini_client.models.generate_content(
        model="gemini-1.5-pro", contents=[prompt]
    )
    return response.text.strip()


def disambiguate_query(original_query: str) -> str:
    """
    Main function to disambiguate the user's query using the ReAct framework.
    """
    print(f"\n[User Query] {original_query}")

    # Assess ambiguity
    prompt = assess_ambiguity_thought(original_query)
    ambiguity = assess_ambiguity_action(prompt)
    print(f"[Observation] Ambiguity assessment: '{ambiguity}'")

    if ambiguity.startswith("no"):
        print("[Action] The query is clear. Proceeding without clarification.")
        return original_query

    # Generate clarifying question
    prompt = generate_clarifying_question_thought(original_query)
    clarifying_question = generate_clarifying_question_action(prompt)
    print(f"[Observation] Clarifying question: '{clarifying_question}'")

    # Get user's response
    user_response = input(f"\n[Agent] {clarifying_question}\n[User] ").strip()
    print(f"[Observation] User's response: '{user_response}'")

    # Formulate final query
    prompt = formulate_final_query_thought(
        original_query, clarifying_question, user_response
    )
    final_query = formulate_final_query_action(prompt)
    print(f"[Observation] Final disambiguated query: '{final_query}'")

    return final_query
