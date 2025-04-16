from fastapi import APIRouter, HTTPException
from app.models.analysis import GaAnalysisRequest, GaAnalysisResponse
from app.core.config import settings
import google.generativeai as genai
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ga", response_model=GaAnalysisResponse)
async def analyze_ga_performance(request: GaAnalysisRequest):
    # Validate API Key
    if not settings.GEMINI_API_KEY:
        logger.error("Gemini API key not configured")
        raise HTTPException(status_code=500, detail="Gemini API not configured")
    
    # Configure Gemini
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Format data for prompt
    data_description = f"""
    Genetic Algorithm Configuration:
    - Generations: {request.generations}
    - Population Size: {request.population_size}
    - Mutation Rate: {request.mutation_rate or 'Not specified'}
    - Mutation Strength: {request.mutation_strength or 'Not specified'}

    Performance Data:
    - Max Fitness History: {request.fitness_history}
    - Avg Fitness History: {request.avg_fitness_history or 'Not provided'}
    - Diversity History: {request.diversity_history or 'Not provided'}
    """
    
    # Construct prompt
    prompt = f"""
    Analyze the following Genetic Algorithm performance data for a machine learning model weight evolution task:

    {data_description}

    Provide detailed analysis covering:
    1. Convergence patterns and generation-wise progress
    2. Diversity trends and exploration/exploitation balance
    3. Effectiveness of mutation parameters
    4. Suggested parameter adjustments
    5. Potential algorithm improvements

    Use Markdown formatting with headings and bullet points.
    """
    
    try:
        response = await model.generate_content_async(prompt)
        return {"analysis_text": response.text}
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis failed")
