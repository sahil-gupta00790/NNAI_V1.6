// lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

interface GeminiHistoryItem {
    role: 'user' | 'model';
    parts: { text: string }[];
}

export async function postGeminiQuery(query: string, history?: GeminiHistoryItem[]) {
    const response = await fetch(`${API_BASE_URL}/gemini/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Send query and history in the expected format
        body: JSON.stringify({ query, history: history || [] }), // Send empty array if no history
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

// New Gemini analysis function
export async function analyzeGaResults(data: {
    fitness_history: number[];
    avg_fitness_history: number[] | null;
    diversity_history: number[] | null;
    generations: number;
    population_size: number;
    mutation_rate?: number;
    mutation_strength?: number;
}): Promise<{ analysis_text: string }> {
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
        throw new Error(`Invalid endpoint type passed to getTaskStatus: ${endpoint}`);
    }
    const response = await fetch(`${API_BASE_URL}/${endpoint}/status/${taskId}`);
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`API Error (${response.status}) fetching status for ${endpoint}: ${errorData.detail || 'Unknown error'}`);
    }
    return response.json();
}

export async function postResetChat() {
    const response = await fetch(`${API_BASE_URL}/advisor/reset_chat`, { method: 'POST' });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`API Error (${response.status}): ${errorData.detail || 'Failed to reset chat'}`);
    }
    return response.json();
}
