// lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export async function postAdvisorQuery(query: string, history?: any[]) {
    const response = await fetch(`${API_BASE_URL}/advisor/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history }),
    });
    // Add more robust error handling
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

export async function getTaskStatus(endpoint: 'evolver', taskId: string) { // Remove 'quantizer' from endpoint type [2]
    // Simplified endpoint logic now that only evolver uses this
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

// Remove startQuantizationTask function entirely [2]
/*
export async function startQuantizationTask(formData: FormData) {
     // ... removed code ...
}
*/

// Optional: Remove reset chat function if not used, or keep it
export async function postResetChat() {
    const response = await fetch(`${API_BASE_URL}/advisor/reset_chat`, { method: 'POST' });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`API Error (${response.status}): ${errorData.detail || 'Failed to reset chat'}`);
    }
    return response.json();
}

