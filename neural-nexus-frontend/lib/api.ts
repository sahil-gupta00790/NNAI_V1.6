// lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// --- Interfaces ---
interface GeminiHistoryItem {
    role: 'user' | 'model';
    parts: { text: string }[];
}

// Interface for the expected structure of the analysis payload
// Includes dynamic rate and hyperparameter fields now
interface GaAnalysisPayload {
    fitness_history: number[];
    avg_fitness_history: number[] | null;
    diversity_history: number[] | null;
    generations: number;
    population_size: number;
    mutation_strength?: number;
    // Fixed Rate
    mutation_rate?: number;
    // Dynamic Rate
    use_dynamic_mutation_rate?: boolean;
    dynamic_mutation_heuristic?: string;
    initial_mutation_rate?: number;
    final_mutation_rate?: number;
    normal_fitness_mutation_rate?: number;
    stagnation_mutation_rate?: number;
    stagnation_threshold?: number;
    base_mutation_rate?: number;
    diversity_threshold_low?: number;
    mutation_rate_increase_factor?: number;
    // Hyperparameters
    evolvable_hyperparams?: Record<string, any> | null;
    best_hyperparameters?: Record<string, any> | null;
}

interface GaAnalysisResponse {
    analysis_text: string;
}

interface TerminateTaskResponse {
    message: string;
    task_id: string;
}
// --- End Interfaces ---


// --- API Functions ---

export async function postGeminiQuery(query: string, history?: GeminiHistoryItem[]) {
    const response = await fetch(`${API_BASE_URL}/gemini/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history: history || [] }),
    });
    if (!response.ok) {
         const errorData = await response.json().catch(() => ({ detail: response.statusText }));
         throw new Error(`Gemini Chat API Error (${response.status}): ${errorData.detail || 'Unknown error'}`);
     }
    return response.json(); // Expects { reply: "..." }
}

export async function postAdvisorQuery(query: string, history?: any[]) {
    const response = await fetch(`${API_BASE_URL}/advisor/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history }),
    });
    if (!response.ok) {
         const errorData = await response.json().catch(() => ({ detail: response.statusText }));
         throw new Error(`API Error (${response.status}): ${errorData.detail || 'Unknown RAG advisor error'}`);
     }
    return response.json();
}

export async function startEvolutionTask(formData: FormData) {
     const response = await fetch(`${API_BASE_URL}/evolver/start`, {
        method: 'POST',
        body: formData,
    });
     if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`API Error (${response.status}): ${errorData.detail || 'Failed to start evolution task'}`);
    }
    return response.json();
}

// Gemini analysis function (updated input type)
export async function analyzeGaResults(data: GaAnalysisPayload): Promise<GaAnalysisResponse> {
    const response = await fetch(`${API_BASE_URL}/analysis/ga`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`Analysis failed: ${errorData.detail || 'Unknown error'}`);
    }
    return response.json();
}

export async function getTaskStatus(endpoint: 'evolver', taskId: string) {
    if (endpoint !== 'evolver') {
        // This basic check can remain, or be expanded if other task types are added
        throw new Error(`Invalid endpoint type passed to getTaskStatus: ${endpoint}`);
    }
    const response = await fetch(`${API_BASE_URL}/${endpoint}/status/${taskId}`);
    // Consider adding more robust error handling if specific status codes mean different things
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        // Distinguish between 404 (task not found) and other errors?
        if (response.status === 404) {
             console.warn(`Task status not found for ${endpoint}/${taskId}. Maybe it hasn't started or ID is wrong.`);
             // Decide how to handle 404 - throw specific error or return null/default state?
             // Throwing for now:
             throw new Error(`Task not found (${response.status}): ${errorData.detail || 'Task ID may be invalid or task not initialized'}`);
        }
        throw new Error(`API Error (${response.status}) fetching status for ${endpoint}: ${errorData.detail || 'Unknown error'}`);
    }
    return response.json(); // Expects the TaskState structure from the backend
}

export async function postResetChat() {
    const response = await fetch(`${API_BASE_URL}/advisor/reset_chat`, { method: 'POST' });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`API Error (${response.status}): ${errorData.detail || 'Failed to reset chat'}`);
    }
    return response.json();
}


// --- NEW Function for Terminating Task ---
/**
 * Sends a request to the backend to terminate a running evolution task.
 * Assumes the backend endpoint is POST /api/v1/evolver/tasks/{taskId}/terminate
 * @param taskId The ID of the task to terminate.
 * @returns Promise resolving to the backend confirmation message.
 */
export async function terminateEvolutionTask(taskId: string): Promise<TerminateTaskResponse> {
    // Construct the URL - Adjust if your backend uses a different structure
    const url = `${API_BASE_URL}/evolver/tasks/${taskId}/terminate`;

    console.log(`Sending termination request for task ${taskId} to ${url}`); // Debug log

    const response = await fetch(url, {
        method: 'POST', // Or 'DELETE' if your backend uses that
        headers: {
            // No 'Content-Type' needed if no body is sent
        },
        // No body needed, taskId is in the URL
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        // Handle specific errors like 404 (task already finished/not found) if needed
        if (response.status === 404) {
             throw new Error(`Task not found or already finished (${response.status}): ${errorData.detail || 'Task may not be running.'}`);
        }
        throw new Error(`API Error (${response.status}) terminating task: ${errorData.detail || 'Unknown error'}`);
    }

    // Expect a confirmation message like {"message": "...", "task_id": "..."}
    return response.json();
}
// --- End NEW Function ---

// --- End API Functions ---
