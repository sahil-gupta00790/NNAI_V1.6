# app/api/endpoints/gemini.py
from fastapi import APIRouter, HTTPException, Body
from app.models.gemini import GeminiChatRequest, GeminiChatResponse, GeminiHistoryItem
from app.core.config import settings
import google.generativeai as genai
import logging
from typing import List

router = APIRouter()
logger = logging.getLogger(__name__)

# Configure Gemini client (should ideally happen once at startup, but simple here)
# Ensure API key is handled securely
if settings.GEMINI_API_KEY:
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
    except Exception as config_err:
        logger.error(f"Failed to configure Gemini API on module load: {config_err}", exc_info=True)
        # Decide how to handle this - maybe raise an error or log prominently
else:
    logger.warning("GEMINI_API_KEY not found in settings. Direct Gemini chat endpoint will fail.")

# Specify the model to use
GEMINI_CHAT_MODEL = 'gemini-2.0-flash' # Or your preferred chat model

@router.post("/chat", response_model=GeminiChatResponse)
async def direct_gemini_chat(request: GeminiChatRequest = Body(...)):
    """Handles direct chat requests with the Gemini API."""

    if not settings.GEMINI_API_KEY:
        logger.error("Attempted direct Gemini chat without API key configured.")
        raise HTTPException(status_code=503, detail="AI Chat service is not configured.")

    # Convert incoming history (if any) to the format expected by start_chat
    # The model expects {'role': ..., 'parts': [{'text': ...}]}
    formatted_history = []
    if request.history:
         formatted_history = [item.model_dump() for item in request.history]
         # Basic validation or transformation if needed

    logger.info(f"Initiating direct chat with model: {GEMINI_CHAT_MODEL}")
    try:
        model = genai.GenerativeModel(GEMINI_CHAT_MODEL)
        # Start chat session with history
        chat = model.start_chat(history=formatted_history)

        logger.info(f"Sending query to Gemini: '{request.query[:50]}...'")
        # Send the new user message
        response = await chat.send_message_async(request.query)
        logger.info("Received response from Gemini.")

        # Basic check for response content
        if not response.text:
            logger.warning("Gemini API returned empty text response.")
            # Check for safety reasons, etc.
            try:
                 feedback = response.prompt_feedback
                 if feedback and feedback.block_reason:
                      logger.warning(f"Gemini response blocked. Reason: {feedback.block_reason_message}")
                      raise HTTPException(status_code=400, detail=f"Response blocked by safety filter: {feedback.block_reason_message}")
            except (AttributeError, ValueError):
                 pass # Ignore if feedback structure is unexpected
            raise HTTPException(status_code=500, detail="AI failed to generate a response.")

        return GeminiChatResponse(reply=response.text)

    except Exception as e:
        logger.error(f"Error during direct Gemini chat: {e}", exc_info=True)
        # Provide a generic error to the user
        error_detail = f"AI chat request failed: {str(e)}"
        # Check for specific Google API error types if needed
        raise HTTPException(status_code=500, detail=error_detail)

