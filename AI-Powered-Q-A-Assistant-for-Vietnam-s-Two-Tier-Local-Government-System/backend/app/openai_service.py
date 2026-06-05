"""
OpenAI integration service — sends conversation context and receives AI responses.
"""

import openai
from loguru import logger
from app.config import get_settings
from app.models import Message

settings = get_settings()

# Configure the OpenAI client
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable AI assistant. "
    "You provide clear, accurate, and well-structured answers. "
    "When appropriate, use markdown formatting for better readability. "
    "Be concise but thorough."
)


def _build_messages(user_message: str, conversation_history: list[Message]) -> list[dict]:
    """
    Build the messages list to send to OpenAI.
    Includes system prompt, conversation history, and the new user message.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (limit to last 20 messages to stay within token limits)
    for msg in conversation_history[-20:]:
        messages.append({"role": msg.role, "content": msg.content})

    # Add the new user message
    messages.append({"role": "user", "content": user_message})

    return messages


def generate_response(user_message: str, conversation_history: list[Message]) -> str:
    """
    Send a message with conversation context to OpenAI and return the response.

    Args:
        user_message: The user's new message.
        conversation_history: Previous messages in the conversation.

    Returns:
        The assistant's response text.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    messages = _build_messages(user_message, conversation_history)

    try:
        logger.info(f"Calling OpenAI API with model={settings.OPENAI_MODEL}, messages_count={len(messages)}")

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            temperature=settings.OPENAI_TEMPERATURE,
        )

        assistant_content = response.choices[0].message.content
        logger.info(f"OpenAI response received: {len(assistant_content)} chars")
        return assistant_content

    except openai.APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        raise Exception("Failed to connect to AI service. Please try again later.")
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit: {e}")
        raise Exception("AI service is currently busy. Please try again in a moment.")
    except openai.APIStatusError as e:
        logger.error(f"OpenAI API error: {e.status_code} — {e.message}")
        raise Exception(f"AI service error: {e.message}")


async def generate_response_stream(user_message: str, conversation_history: list[Message]):
    """
    Stream a response from OpenAI token-by-token.

    Yields:
        String chunks of the assistant's response.
    """
    messages = _build_messages(user_message, conversation_history)

    try:
        logger.info(f"Streaming OpenAI API with model={settings.OPENAI_MODEL}")

        stream = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            temperature=settings.OPENAI_TEMPERATURE,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except Exception as e:
        logger.error(f"OpenAI streaming error: {e}")
        yield f"\n\n[Error: {str(e)}]"
