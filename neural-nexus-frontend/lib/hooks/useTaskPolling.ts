// lib/hooks/useTaskPolling.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import { getTaskStatus } from '@/lib/api'; // Assuming this API function exists
import { toast } from "sonner";

// --- Type Definitions ---
// Add 'HALTED' as a possible status
type TaskStatusType = 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'REVOKED' | 'HALTED' | string;

// Matches the structure returned in the 'result' or 'meta' field of Celery task state
interface TaskResultData {
    final_model_path?: string;
    best_fitness?: number | null;
    message?: string;
    error?: string;
    // Add fields for the new metric histories
    fitness_history?: number[] | null; // Max fitness history
    avg_fitness_history?: number[] | null;
    diversity_history?: number[] | null;
    // Added based on task return values
    best_hyperparameters?: Record<string, any> | null;
    status?: string; // Could be included in result/meta, e.g., HALTED_BY_USER
}

// Structure expected from the getTaskStatus API endpoint (adjust if your backend sends differently)
interface TaskStatusResponse {
    task_id: string;
    status: TaskStatusType;
    progress?: number | null;
    info?: TaskResultData | any | null; // Meta data during PROGRESS/REVOKED/HALTED state
    result?: TaskResultData | any | null; // Final result data on SUCCESS/FAILURE
    message?: string | null;
}

// State managed by the hook
interface TaskState {
    taskId: string | null;
    status: TaskStatusType | null;
    progress: number | null;
    result: TaskResultData | null;
    fitnessHistory: number[] | null;
    avgFitnessHistory: number[] | null;
    diversityHistory: number[] | null;
    message: string | null;
    error: string | null;
    isActive: boolean;
}

const initialState: TaskState = {
    taskId: null, status: null, progress: null, result: null,
    fitnessHistory: null, avgFitnessHistory: null, diversityHistory: null,
    message: null, error: null, isActive: false
};

// Terminal states that should stop polling
const TERMINAL_STATES: TaskStatusType[] = ['SUCCESS', 'FAILURE', 'REVOKED', 'HALTED'];

export function useTaskPolling(endpoint: 'evolver', intervalMs = 3000) {
    const [taskState, setTaskState] = useState<TaskState>(initialState);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const currentTaskIdRef = useRef<string | null>(null);

    // Function to stop polling interval
    const stopPolling = useCallback(() => {
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
        // Ensure isActive reflects the terminal state correctly
        setTaskState(prev => {
            // If previously active AND not already in a known terminal state, set inactive.
            // If already in a terminal state, keep it as inactive.
            if (prev.isActive && !TERMINAL_STATES.includes(prev.status || '')) {
                return { ...prev, isActive: false };
            }
            return prev; // Otherwise, keep state as is
        });
        currentTaskIdRef.current = null; // Clear tracked task ID
    }, []); // No dependencies needed for this version


    // Function to poll status API
    const pollStatus = useCallback(async () => {
        const taskId = currentTaskIdRef.current;
        if (!taskId) { stopPolling(); return; }

        try {
            const data: TaskStatusResponse = await getTaskStatus('evolver', taskId);

            setTaskState(prev => {
                if (prev.taskId !== taskId) return prev; // Ignore if task changed

                // Determine terminal state
                const isTerminal = TERMINAL_STATES.includes(data.status);

                // Determine the source of metadata based on status
                // Use 'info' for intermediate/halted states, 'result' for final SUCCESS/FAILURE
                // Fallback to an empty object if neither exists
                const metaSource = (data.status === 'PROGRESS' || data.status === 'REVOKED' || data.status === 'HALTED')
                    ? data.info
                    : (data.status === 'SUCCESS' || data.status === 'FAILURE'
                        ? data.result
                        : null)
                    || {}; // Ensure metaSource is always an object or null

                const finalResultData = (data.status === 'SUCCESS' || data.status === 'FAILURE') ? data.result : null;

                // Extract Histories safely from the metaSource (could be info or result)
                const nextFitnessHistory = metaSource?.fitness_history && Array.isArray(metaSource.fitness_history) ? metaSource.fitness_history : prev.fitnessHistory;
                const nextAvgFitnessHistory = metaSource?.avg_fitness_history && Array.isArray(metaSource.avg_fitness_history) ? metaSource.avg_fitness_history : prev.avgFitnessHistory;
                const nextDiversityHistory = metaSource?.diversity_history && Array.isArray(metaSource.diversity_history) ? metaSource.diversity_history : prev.diversityHistory;

                // Extract Progress (usually in 'info' during PROGRESS)
                const nextProgress = (data.status === 'PROGRESS' && typeof metaSource?.progress === 'number') ? metaSource.progress : (isTerminal ? 1.0 : prev.progress); // Set to 1 on any terminal state

                // Extract Message (Prioritize metaSource message, then top-level message)
                const nextMessage = metaSource?.message || data.message || prev.message;

                // Extract Error (usually in 'result' on FAILURE)
                const nextError = data.status === 'FAILURE' ? (finalResultData?.error || finalResultData?.message || 'Task failed') : null;

                const newState: TaskState = {
                    ...prev,
                    status: data.status,
                    progress: nextProgress,
                    // Store final result for SUCCESS/FAILURE, store meta for HALTED/REVOKED for consistency
                    result: (data.status === 'SUCCESS' || data.status === 'FAILURE') ? finalResultData : (data.status === 'REVOKED' || data.status === 'HALTED' ? metaSource : null),
                    fitnessHistory: nextFitnessHistory,
                    avgFitnessHistory: nextAvgFitnessHistory,
                    diversityHistory: nextDiversityHistory,
                    message: nextMessage,
                    error: nextError,
                    isActive: !isTerminal, // Set isActive based on whether it's a terminal state
                };

                // Handle Side Effects (Toasts) - outside state update logic
                if (isTerminal && prev.status !== data.status) { // Only toast on final state *change*
                    if (data.status === 'SUCCESS') toast.success(`Task ${taskId} completed!`);
                    else if (data.status === 'FAILURE') toast.error(`Task ${taskId} failed: ${newState.error || 'Unknown error'}`);
                    else if (data.status === 'REVOKED') toast.warning(`Task ${taskId} was revoked.`);
                    else if (data.status === 'HALTED') toast.info(`Task ${taskId} was halted.`); // Add toast for HALTED
                }
                return newState;
            });

            // Stop interval *after* state update if a terminal state was reached
             if (TERMINAL_STATES.includes(data.status)) {
                 stopPolling();
             }

        } catch (error: any) {
            console.error(`Polling error for task ${taskId}:`, error);
            const errorMsg = error.message || 'Polling request failed';
            setTaskState(prev => ({ ...prev, error: errorMsg, isActive: false })); // Set inactive on polling error
            toast.error(`Polling Error: ${errorMsg}`);
            stopPolling(); // Stop polling on error
        }
    }, [stopPolling, intervalMs]); // Dependencies for pollStatus

    // Function to start a new task and polling
    const startTask = useCallback((taskId: string) => {
        stopPolling(); // Ensure any previous polling is stopped
        currentTaskIdRef.current = taskId;
        setTaskState(initialState); // Reset to initial state FIRST
        setTaskState(prev => ({ // Then set the new task ID and pending status
            ...prev,
            taskId: taskId,
            status: 'PENDING',
            isActive: true,
            message: 'Task submitted, waiting for status...'
        }));

        // Immediate first poll after a short delay
        const firstPollTimeout = setTimeout(() => { if (currentTaskIdRef.current === taskId) { pollStatus(); } }, 1000); // Reduced delay

        // Setup interval
        if (intervalRef.current) clearInterval(intervalRef.current); // Clear just in case
        intervalRef.current = setInterval(pollStatus, intervalMs);

        // Cleanup timeout if component unmounts before it fires
        // Note: This specific cleanup might be redundant due to stopPolling in unmount effect
        // return () => clearTimeout(firstPollTimeout);

    }, [stopPolling, pollStatus, intervalMs]); // Dependencies for startTask

    // Cleanup interval on component unmount
    useEffect(() => {
        // Return the cleanup function
        return () => {
            stopPolling();
        };
    }, [stopPolling]); // Dependency: stopPolling

    // Function to manually reset state (e.g., after halting or error)
    const resetTaskState = useCallback(() => {
        stopPolling();
        setTaskState(initialState);
    }, [stopPolling]);

    // Return state and control functions
    return { taskState, startTask, resetTaskState }; // Removed stopPolling from return unless needed externally
}
