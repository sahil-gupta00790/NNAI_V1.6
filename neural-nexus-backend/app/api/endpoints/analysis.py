# app/api/endpoints/analysis.py

from fastapi import APIRouter, HTTPException, Body # Added Body for potential debugging
from app.models.analysis import GaAnalysisRequest, GaAnalysisResponse # Ensure path is correct
from app.core.config import settings
import google.generativeai as genai
import logging
from typing import Dict, Any # For type hinting

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ga", response_model=GaAnalysisResponse)
async def analyze_ga_performance(
    request: GaAnalysisRequest # Use the updated request model
    # request: Dict[str, Any] = Body(...) # Use this temporarily for debugging if Pydantic fails
):
    # --- Debugging incoming request (Optional) ---
    # logger.info(f"Received analysis request body type: {type(request)}")
    # logger.info(f"Received analysis request data: {request}")
    # If using Dict for debug: request_obj = GaAnalysisRequest(**request)
    # else:
    request_obj = request
    # --- End Debugging ---


    # Validate API Key
    if not settings.GEMINI_API_KEY:
        logger.error("Gemini API key not configured")
        raise HTTPException(status_code=500, detail="Gemini API not configured")

    # Configure Gemini
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Consider making the model configurable (e.g., gemini-2.0-flash)
        model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info(f"Using Gemini model: {model.model_name}")
    except Exception as config_err:
        logger.error(f"Failed to configure Gemini: {config_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gemini configuration error")


    # --- Build Dynamic Configuration Description ---
    config_lines = [
        f"- Generations: {request_obj.generations}",
        f"- Population Size: {request_obj.population_size}",
        f"- Mutation Strength (Weights): {request_obj.mutation_strength if request_obj.mutation_strength is not None else 'Not specified'}"
    ]

    if request_obj.use_dynamic_mutation_rate:
        config_lines.append("- Dynamic Weight Mutation Rate: ENABLED")
        heuristic = request_obj.dynamic_mutation_heuristic or 'Unknown'
        config_lines.append(f"  - Strategy: {heuristic.replace('_', ' ').title()}")
        if heuristic == 'time_decay':
            config_lines.append(f"    - Initial Rate: {request_obj.initial_mutation_rate if request_obj.initial_mutation_rate is not None else 'N/A'}")
            config_lines.append(f"    - Final Rate: {request_obj.final_mutation_rate if request_obj.final_mutation_rate is not None else 'N/A'}")
        elif heuristic == 'fitness_based':
            config_lines.append(f"    - Normal Rate (Improving): {request_obj.normal_fitness_mutation_rate if request_obj.normal_fitness_mutation_rate is not None else 'N/A'}")
            config_lines.append(f"    - Increased Rate (Stagnation): {request_obj.stagnation_mutation_rate if request_obj.stagnation_mutation_rate is not None else 'N/A'}")
            config_lines.append(f"    - Stagnation Threshold: {request_obj.stagnation_threshold if request_obj.stagnation_threshold is not None else 'N/A'}")
        elif heuristic == 'diversity_based':
            config_lines.append(f"    - Base Rate: {request_obj.base_mutation_rate if request_obj.base_mutation_rate is not None else 'N/A'}")
            config_lines.append(f"    - Low Diversity Threshold: {request_obj.diversity_threshold_low if request_obj.diversity_threshold_low is not None else 'N/A'}")
            config_lines.append(f"    - Rate Increase Factor: {request_obj.mutation_rate_increase_factor if request_obj.mutation_rate_increase_factor is not None else 'N/A'}")
    else:
        config_lines.append("- Dynamic Weight Mutation Rate: DISABLED")
        config_lines.append(f"  - Fixed Rate: {request_obj.mutation_rate if request_obj.mutation_rate is not None else 'Not specified'}")

    # Add Hyperparameter Optimization Info
    if request_obj.evolvable_hyperparams and request_obj.best_hyperparameters:
        evolved_keys = list(request_obj.evolvable_hyperparams.keys())
        if evolved_keys:
            config_lines.append(f"- Hyperparameter Optimization: ENABLED")
            config_lines.append(f"  - Evolved Parameters: {', '.join(evolved_keys)}")
            # Format best hyperparameters nicely
            best_hparams_str = ', '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in request_obj.best_hyperparameters.items()])
            config_lines.append(f"  - Best Found Set: {{ {best_hparams_str} }}")
    # --- End Building Config Description ---

    # Format data for prompt
    # Truncate long histories to keep prompt manageable, keep start/end?
    max_history_len = 50 # Example limit
    truncated_fitness = request_obj.fitness_history[:max_history_len] + (["..."] if len(request_obj.fitness_history) > max_history_len else [])
    truncated_avg_fitness = (request_obj.avg_fitness_history[:max_history_len] + (["..."] if request_obj.avg_fitness_history and len(request_obj.avg_fitness_history) > max_history_len else []) if request_obj.avg_fitness_history else ["Not provided"])
    truncated_diversity = (request_obj.diversity_history[:max_history_len] + (["..."] if request_obj.diversity_history and len(request_obj.diversity_history) > max_history_len else []) if request_obj.diversity_history else ["Not provided"])

    data_description = f"""
    Genetic Algorithm Configuration:
{chr(10).join(config_lines)}

    Performance Data (potentially truncated):
    - Max Fitness History: {truncated_fitness}
    - Avg Fitness History: {truncated_avg_fitness}
    - Diversity History: {truncated_diversity}
    """

    # --- Construct Updated Prompt ---
    # Added specific instructions for dynamic rates and hyperparameter tuning
    prompt_instructions = [
        "Analyze the following Genetic Algorithm (GA) performance data for a machine learning model evolution task.",
        "Provide detailed analysis using Markdown formatting (headings, lists). Cover these aspects:",
        "1. **Convergence & Progress:** Describe the overall fitness trend (max and average). Did it converge? Were there distinct phases (e.g., rapid initial improvement, plateau)?",
        "2. **Exploration vs. Exploitation:** Analyze the diversity history. Did the population maintain diversity, or did it collapse? How does this correlate with fitness progress? Does it suggest a good balance?",
        "3. **Mutation Parameter Effectiveness:**"
    ]
    # Add specific mutation instructions
    if request_obj.use_dynamic_mutation_rate:
        prompt_instructions.append(f"   - Evaluate the **dynamic weight mutation rate** strategy ('{request_obj.dynamic_mutation_heuristic or 'Unknown'}'). Based on the fitness/diversity curves, did this strategy seem effective in balancing exploration/exploitation compared to what might be expected from a fixed rate? (e.g., did diversity recover during stagnation if fitness/diversity based? did decay match convergence?)")
    else:
        prompt_instructions.append(f"   - Evaluate the **fixed weight mutation rate** ({request_obj.mutation_rate if request_obj.mutation_rate is not None else 'N/A'}) and strength ({request_obj.mutation_strength if request_obj.mutation_strength is not None else 'N/A'}). Were they appropriate? Too high (preventing convergence)? Too low (causing stagnation)?")

    # Add hyperparameter instructions
    if request_obj.evolvable_hyperparams and request_obj.best_hyperparameters and list(request_obj.evolvable_hyperparams.keys()):
        prompt_instructions.append("4. **Hyperparameter Optimization:**")
        prompt_instructions.append(f"   - Consider that hyperparameters ({', '.join(list(request_obj.evolvable_hyperparams.keys()))}) were co-evolved with weights. Does the fitness progression suggest this was beneficial? Were there periods where hyperparameter changes might have driven fitness improvements?")
        prompt_instructions.append(f"   - The best hyperparameter set found was approximately: {request_obj.best_hyperparameters}.")
        prompt_instructions.append("5. **Suggestions & Improvements:** Based on the analysis, suggest specific adjustments to GA parameters (population size, selection, crossover, mutation settings) or potential algorithmic changes for future runs on similar tasks.")
    else:
         prompt_instructions.append("4. **Suggestions & Improvements:** Based on the analysis, suggest specific adjustments to GA parameters (population size, selection, crossover, mutation settings) or potential algorithmic changes for future runs on similar tasks.")

    # Combine into final prompt
    prompt = f"""
{chr(10).join(prompt_instructions)}

    --- Data ---
    {data_description}
    --- End Data ---

    Begin Analysis:
    """
    # --- End Prompt Construction ---

    logger.debug(f"Sending prompt to Gemini:\n{prompt[:500]}...") # Log start of prompt

    try:
        # Use async client for FastAPI endpoint
        response = await model.generate_content_async(prompt)

        # Basic validation of response structure (Gemini API specific)
        if not response or not hasattr(response, 'text'):
             logger.error(f"Gemini response invalid or missing 'text' attribute. Response: {response}")
             raise HTTPException(status_code=500, detail="Received invalid response from analysis model.")

        analysis_text = response.text
        logger.info("Gemini analysis generated successfully.")
        return {"analysis_text": analysis_text}

    except Exception as e:
        # Catch potential Gemini API errors (e.g., rate limits, content filtering)
        logger.error(f"Gemini API call failed: {str(e)}", exc_info=True)
        # Provide a more specific error message if possible
        error_detail = f"Analysis failed due to API error: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)

