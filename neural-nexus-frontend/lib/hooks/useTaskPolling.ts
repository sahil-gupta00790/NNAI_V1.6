// lib/hooks/useTaskPolling.ts
import { useState, useEffect, useRef, useCallback } from 'react'; // Added useCallback
import { getTaskStatus } from '@/lib/api';
import { toast } from "sonner";

// Type definitions
type TaskStatusType = 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'REVOKED' | string;

interface TaskResultData {
    final_model_path?: string; // Make specific if possible
    error?: string;
    [key: string]: any; // Allow other potential result fields
}

interface TaskStatusResponse {
    task_id: string;
    status: TaskStatusType;
    progress: number | null;
    result: {
        data: TaskResultData | null; // Contains actual result data or error info
        fitness_history?: number[] | null; // Optional fitness history
    } | null;
    message: string | null;
}

interface TaskState {
    taskId: string | null;
    status: TaskStatusType | null;
    progress: number | null;
    result: TaskResultData | null; // Store just the 'data' part of the result
    fitnessHistory: number[] | null;
    message: string | null;
    error: string | null;
    isActive: boolean;
}

const initialState: TaskState = {
    taskId: null, status: null, progress: null, result: null, fitnessHistory: null, message: null, error: null, isActive: false
};

// Update function signature to only accept 'evolver'
export function useTaskPolling(endpoint: 'evolver', intervalMs = 3000) {
    const [taskState, setTaskState] = useState<TaskState>(initialState);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const currentTaskIdRef = useRef<string | null>(null); // Ref to store the current active task ID

    const stopPolling = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        // Only set isActive false if there isn't an error or final state already set
        setTaskState(prev => {
            if (prev.status !== 'SUCCESS' && prev.status !== 'FAILURE' && prev.status !== 'REVOKED' && !prev.error) {
                return { ...prev, isActive: false };
            }
            return prev; // Keep state as is if already completed/failed
        });
        currentTaskIdRef.current = null; // Clear the active task ID ref
    }, []); // No dependencies, function definition is stable


    const pollStatus = useCallback(async () => {
        const taskId = currentTaskIdRef.current; // Get task ID from ref
        if (!taskId) {
            // console.warn("pollStatus called without active taskId");
            stopPolling();
            return;
        }

        // console.log(`Polling for task: ${taskId}`); // Debug log

        try {
            // Call getTaskStatus directly with 'evolver'
            const data: TaskStatusResponse = await getTaskStatus('evolver', taskId);

            // console.log("Received status data:", data); // Debug log

            // Use functional update to ensure we work with the latest state
            setTaskState(prev => {
                // If the task ID changed while polling was in flight, ignore this update
                if (prev.taskId !== taskId) return prev;

                // Determine next fitness history based on previous state and new data
                let nextFitnessHistory = prev.fitnessHistory; // Default to previous
                if (data.result?.fitness_history && Array.isArray(data.result.fitness_history)) {
                    nextFitnessHistory = data.result.fitness_history; // Use new data if available
                }

                // Determine if polling should stop
                const shouldStop = data.status === 'SUCCESS' || data.status === 'FAILURE' || data.status === 'REVOKED';

                // Calculate new state
                const newState: TaskState = {
                    ...prev,
                    status: data.status,
                    progress: data.progress,
                    // Store only the nested 'data' object from the result, or null
                    result: data.result?.data ?? null,
                    fitnessHistory: nextFitnessHistory,
                    message: data.message,
                    // Extract error message preferably from result.data.error
                    error: data.status === 'FAILURE' ? (data.result?.data?.error || data.message || 'Task failed') : null,
                    // isActive should become false ONLY when stopping
                    isActive: !shouldStop,
                };

                // --- Side Effects (Toasts, Stopping Interval) ---
                // Need to be handled carefully after state update is calculated
                // We can't call stopPolling directly inside setTaskState
                if (shouldStop) {
                    // console.log(`Stopping polling for ${taskId}, status: ${data.status}`); // Debug log
                    // Trigger toast notifications based on the final status derived *now*
                    if (data.status === 'SUCCESS') {
                       toast.success(`Task ${taskId} completed!`);
                    } else if (data.status === 'FAILURE') {
                       const errorMsg = data.result?.data?.error || data.message || 'Unknown error';
                       toast.error(`Task ${taskId} failed: ${errorMsg}`);
                    } else if (data.status === 'REVOKED') {
                       toast.info(`Task ${taskId} was revoked.`);
                    }
                }

                return newState;
            });

            // If the status indicates stopping, clear the interval *after* the state update
             if (data.status === 'SUCCESS' || data.status === 'FAILURE' || data.status === 'REVOKED') {
                 stopPolling(); // Call stopPolling here
             }

        } catch (error: any) {
            console.error(`Polling error for task ${taskId}:`, error);
            const errorMsg = error.message || 'Polling request failed';
            setTaskState(prev => ({ ...prev, error: errorMsg, isActive: false }));
            toast.error(`Polling Error: ${errorMsg}`);
            stopPolling(); // Stop interval on catch
        }
    }, [stopPolling]); // Include stopPolling as a dependency (it's memoized by useCallback)


    const startTask = useCallback((taskId: string) => {
        // console.log(`Starting task: ${taskId}`); // Debug log
        stopPolling(); // Stop any previous polling
        currentTaskIdRef.current = taskId; // Set the active task ID in the ref
        setTaskState({ // Set initial state
            taskId: taskId, status: 'PENDING', progress: 0, result: null, fitnessHistory: null,
            message: 'Task submitted, waiting for status...', error: null, isActive: true,
        });

        // Immediate first poll after a short delay
        const firstPollTimeout = setTimeout(() => {
             // Check ref directly - avoids stale closure on taskState
             if (currentTaskIdRef.current === taskId) {
                 pollStatus();
             }
        }, 1000); // Poll 1 second after starting

        // Set up the interval - ensure the function reference is stable
        intervalRef.current = setInterval(pollStatus, intervalMs);

        // Cleanup function for the timeout (though less critical now)
        // return () => clearTimeout(firstPollTimeout);
    }, [stopPolling, pollStatus, intervalMs]); // Dependencies for useCallback


    // Cleanup interval on component unmount
    useEffect(() => {
        return () => {
            stopPolling();
        };
    }, [stopPolling]); // Dependency on memoized stopPolling


    const resetTaskState = useCallback(() => {
        // console.log("Resetting task state"); // Debug log
        stopPolling();
        setTaskState(initialState); // Reset to initial state object
    }, [stopPolling]); // Dependency on memoized stopPolling

    return { taskState, startTask, stopPolling, resetTaskState };
}
