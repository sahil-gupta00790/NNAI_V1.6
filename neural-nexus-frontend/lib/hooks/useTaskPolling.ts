// lib/hooks/useTaskPolling.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import { getTaskStatus } from '@/lib/api'; // Assuming this API function exists
import { toast } from "sonner";

// --- Type Definitions ---
type TaskStatusType = 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'REVOKED' | string;

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
}

// Structure expected from the getTaskStatus API endpoint (adjust if your backend sends differently)
// Celery's AsyncResult often has status, info (for meta), result (for final data)
interface TaskStatusResponse {
    task_id: string;
    status: TaskStatusType;
    progress?: number | null; // Often not directly available, calculate from info if needed
    info?: TaskResultData | any | null; // Meta data during PROGRESS state
    result?: TaskResultData | any | null; // Final result data on SUCCESS/FAILURE
    message?: string | null; // Top-level message might exist
}

// State managed by the hook
interface TaskState {
    taskId: string | null;
    status: TaskStatusType | null;
    progress: number | null;
    result: TaskResultData | null; // Store processed final result data
    // Store metric histories
    fitnessHistory: number[] | null; // Max fitness
    avgFitnessHistory: number[] | null;
    diversityHistory: number[] | null;
    // ---
    message: string | null;
    error: string | null;
    isActive: boolean;
}

const initialState: TaskState = {
    taskId: null, status: null, progress: null, result: null,
    fitnessHistory: null, avgFitnessHistory: null, diversityHistory: null, // Initialize new states
    message: null, error: null, isActive: false
};

// Only 'evolver' needed for endpoint type
export function useTaskPolling(endpoint: 'evolver', intervalMs = 3000) {
    const [taskState, setTaskState] = useState<TaskState>(initialState);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const currentTaskIdRef = useRef<string | null>(null);

    const stopPolling = useCallback(() => {
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
        setTaskState(prev => {
            if (prev.isActive && prev.status !== 'SUCCESS' && prev.status !== 'FAILURE' && prev.status !== 'REVOKED') {
                // Only set isActive to false if it was previously active and not in a final state
                return { ...prev, isActive: false };
            }
            return prev; // Otherwise, keep state as is
        });
        currentTaskIdRef.current = null;
    }, []);


    const pollStatus = useCallback(async () => {
        const taskId = currentTaskIdRef.current;
        if (!taskId) { stopPolling(); return; }

        try {
            const data: TaskStatusResponse = await getTaskStatus('evolver', taskId);

            setTaskState(prev => {
                if (prev.taskId !== taskId) return prev; // Ignore if task changed

                // Determine the source of metadata based on status
                // PROGRESS meta usually in 'info', final SUCCESS/FAILURE data in 'result'
                const metaSource = (data.status === 'PROGRESS' ? data.info : data.result) || {};
                const finalResultData = (data.status === 'SUCCESS' || data.status === 'FAILURE') ? data.result : null;

                // Extract Histories safely from the metaSource
                const nextFitnessHistory = metaSource?.fitness_history && Array.isArray(metaSource.fitness_history)
                    ? metaSource.fitness_history : prev.fitnessHistory;
                const nextAvgFitnessHistory = metaSource?.avg_fitness_history && Array.isArray(metaSource.avg_fitness_history)
                    ? metaSource.avg_fitness_history : prev.avgFitnessHistory;
                const nextDiversityHistory = metaSource?.diversity_history && Array.isArray(metaSource.diversity_history)
                    ? metaSource.diversity_history : prev.diversityHistory;

                // Extract Progress (often in 'info' during PROGRESS)
                const nextProgress = (data.status === 'PROGRESS' && typeof metaSource?.progress === 'number')
                    ? metaSource.progress
                    : (data.status === 'SUCCESS' ? 1.0 : prev.progress); // Set to 1 on success

                // Extract Message
                const nextMessage = metaSource?.message || data.message || prev.message;

                // Extract Error (usually in 'result' on FAILURE)
                const nextError = data.status === 'FAILURE' ? (finalResultData?.error || finalResultData?.message || 'Task failed') : null;

                const shouldStop = data.status === 'SUCCESS' || data.status === 'FAILURE' || data.status === 'REVOKED';

                const newState: TaskState = {
                    ...prev,
                    status: data.status,
                    progress: nextProgress,
                    result: finalResultData, // Store final result data
                    fitnessHistory: nextFitnessHistory,
                    avgFitnessHistory: nextAvgFitnessHistory,
                    diversityHistory: nextDiversityHistory,
                    message: nextMessage,
                    error: nextError,
                    isActive: !shouldStop,
                };

                // Handle Side Effects (Toasts) outside the state updater logic
                if (shouldStop && prev.status !== data.status) { // Only toast on final state change
                    if (data.status === 'SUCCESS') toast.success(`Task ${taskId} completed!`);
                    else if (data.status === 'FAILURE') toast.error(`Task ${taskId} failed: ${newState.error || 'Unknown error'}`);
                    else if (data.status === 'REVOKED') toast.info(`Task ${taskId} was revoked.`);
                }
                return newState;
            });

            // Stop interval *after* state update if needed
             if (data.status === 'SUCCESS' || data.status === 'FAILURE' || data.status === 'REVOKED') {
                 stopPolling();
             }

        } catch (error: any) {
            console.error(`Polling error for task ${taskId}:`, error);
            const errorMsg = error.message || 'Polling request failed';
            setTaskState(prev => ({ ...prev, error: errorMsg, isActive: false }));
            toast.error(`Polling Error: ${errorMsg}`);
            stopPolling();
        }
    }, [stopPolling, intervalMs]); // intervalMs is stable, stopPolling is memoized


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

        // Immediate first poll
        const firstPollTimeout = setTimeout(() => { if (currentTaskIdRef.current === taskId) { pollStatus(); } }, 1000);

        // Setup interval
        intervalRef.current = setInterval(pollStatus, intervalMs);

        // No need to return cleanup from here, useEffect handles unmount cleanup
    }, [stopPolling, pollStatus, intervalMs]); // Dependencies


    // Cleanup interval on component unmount
    useEffect(() => { return () => { stopPolling(); }; }, [stopPolling]);

    const resetTaskState = useCallback(() => {
        stopPolling();
        setTaskState(initialState);
    }, [stopPolling]);

    return { taskState, startTask, stopPolling, resetTaskState };
}
