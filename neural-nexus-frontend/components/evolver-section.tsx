// components/evolver-section.tsx
'use client';
import React, { useState, useRef, ChangeEvent, FormEvent } from 'react'; // Added types
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea'; // *** ADDED Textarea import ***
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group" // For eval choice
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"; // Import Alert components
import { AlertCircle } from "lucide-react"; // Import icon for error alert
import { startEvolutionTask } from '@/lib/api';
import { useTaskPolling } from '@/lib/hooks/useTaskPolling';
import RealTimePlot from './real-time-plot';
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";

// Default JSON structure to guide the user
const defaultJsonConfig = JSON.stringify(
  {
    model_class: "MyCNN", // Instruct user to change this
    generations: 10,
    population_size: 20,
    mutation_rate: 0.1,
    mutation_strength: 0.05,
    // Add other parameters your backend/task expects
  },
  null, // Replacer function for JSON.stringify
  2   // Indentation (spaces) for readability
);


export default function EvolverSection() {
    const [modelDefFile, setModelDefFile] = useState<File | null>(null);
    const [taskEvalFile, setTaskEvalFile] = useState<File | null>(null);
    const [weightsFile, setWeightsFile] = useState<File | null>(null);
    // *** ADDED state for JSON config ***
    const [configJson, setConfigJson] = useState<string>(defaultJsonConfig);
    // Keep generations/popSize state if UI inputs should potentially override JSON (requires backend logic)
    const [generations, setGenerations] = useState<number>(10); // Default generations
    const [popSize, setPopSize] = useState<number>(20); // Default pop size
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [evalChoice, setEvalChoice] = useState<'standard' | 'custom'>('standard'); // Default to standard

    const modelDefRef = useRef<HTMLInputElement>(null);
    const taskEvalRef = useRef<HTMLInputElement>(null);
    const weightsRef = useRef<HTMLInputElement>(null);

    // Use the provided hook for polling
    const { taskState, startTask, resetTaskState } = useTaskPolling('evolver');

    const handleEvalChoiceChange = (value: 'standard' | 'custom') => {
        setEvalChoice(value);
        // Clear custom file if switching back to standard
        if (value === 'standard' && taskEvalRef.current) {
            taskEvalRef.current.value = "";
            setTaskEvalFile(null);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        // Validation
        if (!modelDefFile) {
            toast.error("Model Definition file is required.");
            return;
        }
        if (evalChoice === 'custom' && !taskEvalFile) {
             toast.error("Custom Evaluation Script file is required when selected.");
             return;
        }
        if (taskState.isActive) {
            toast.warning("A task is already running.");
            return;
        }

        // *** ADDED JSON Validation ***
        let parsedConfig;
        try {
            parsedConfig = JSON.parse(configJson);
            if (!parsedConfig.model_class || typeof parsedConfig.model_class !== 'string' || parsedConfig.model_class.trim() === "") {
                // Check if model_class exists, is a non-empty string
                throw new Error("'model_class' key (with a valid class name string) is missing or invalid in Configuration JSON.");
            }
        } catch (jsonError: any) {
             toast.error(`Invalid Configuration JSON: ${jsonError.message}`);
             return; // Stop submission if JSON is invalid
        }
        // *** END JSON Validation ***

        setIsSubmitting(true);
        resetTaskState(); // Clear previous task state before starting new one
        toast("Submitting evolution task...");

        const formData = new FormData();
        formData.append('model_definition', modelDefFile);
        formData.append('use_standard_eval', String(evalChoice === 'standard')); // Send as string 'true'/'false'

        if (evalChoice === 'custom' && taskEvalFile) {
            formData.append('task_evaluation', taskEvalFile);
        }
        if (weightsFile) {
            formData.append('initial_weights', weightsFile);
        }

        // --- Configuration ---
        // Append the validated JSON string from the state
        formData.append('config_json', configJson);
        // --- End Configuration ---

        try {
            const response = await startEvolutionTask(formData);
            startTask(response.task_id); // Start polling for the new task
            toast.success(`Task ${response.task_id} started.`);
             // Reset form fields visually ONLY on successful submission
             if(modelDefRef.current) modelDefRef.current.value = "";
             if(taskEvalRef.current) taskEvalRef.current.value = "";
             if(weightsRef.current) weightsRef.current.value = "";
             setModelDefFile(null);
             setTaskEvalFile(null);
             setWeightsFile(null);
             // Optionally reset config inputs too
             // setConfigJson(defaultJsonConfig);
             // setGenerations(10);
             // setPopSize(20);

        } catch (error: any) {
            console.error("Error starting evolution task:", error);
            toast.error(`Failed to start task: ${error.message || 'Unknown error'}`);
            // Do not reset form on error
        } finally {
            setIsSubmitting(false);
        }
    };

    // --- Prepare plot data and download link (remains the same) ---
    const fitnessHistory: number[] = taskState.fitnessHistory ?? [];
    const downloadLink = taskState.status === 'SUCCESS' && taskState.taskId
        ? `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'}/evolver/results/${taskState.taskId}/download`
        : null;

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Form Card */}
            <Card>
                <CardHeader>
                    <CardTitle>Configure Evolution</CardTitle>
                    <CardDescription>Upload files and set parameters for the Genetic Algorithm.</CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {/* Model Definition */}
                        <div>
                            <Label htmlFor="model-def">Model Definition (.py) <span className="text-red-500">*</span></Label>
                            <Input ref={modelDefRef} id="model-def" type="file" accept=".py" required onChange={(e) => setModelDefFile(e.target.files?.[0] ?? null)} disabled={isSubmitting || taskState.isActive} />
                        </div>

                        {/* Evaluation Choice */}
                        <div>
                             <Label>Evaluation Method <span className="text-red-500">*</span></Label>
                              <RadioGroup
                                value={evalChoice}
                                onValueChange={handleEvalChoiceChange}
                                className="flex space-x-4 mt-1"
                                disabled={isSubmitting || taskState.isActive}
                              >
                                <div className="flex items-center space-x-2">
                                  <RadioGroupItem value="standard" id="eval-standard" />
                                  <Label htmlFor="eval-standard">Standard (MNIST)</Label>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <RadioGroupItem value="custom" id="eval-custom" />
                                  <Label htmlFor="eval-custom">Upload Custom</Label>
                                </div>
                              </RadioGroup>
                        </div>

                         {/* Custom Evaluation Script Input (Conditional) */}
                         <AnimatePresence>
                            {evalChoice === 'custom' && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    transition={{ duration: 0.3 }}
                                    className="mt-2"
                                >
                                    <Label htmlFor="task-eval">Custom Evaluation Script (.py) <span className="text-red-500">*</span></Label>
                                    <Input ref={taskEvalRef} id="task-eval" type="file" accept=".py" required={evalChoice === 'custom'} onChange={(e) => setTaskEvalFile(e.target.files?.[0] ?? null)} disabled={isSubmitting || taskState.isActive} />
                                </motion.div>
                            )}
                        </AnimatePresence>

                         {/* Initial Weights */}
                        <div>
                            <Label htmlFor="init-weights">Initial Weights (.pth, Optional)</Label>
                            <Input ref={weightsRef} id="init-weights" type="file" accept=".pth,.pt" onChange={(e) => setWeightsFile(e.target.files?.[0] ?? null)} disabled={isSubmitting || taskState.isActive} />
                        </div>

                        {/* === ADDED Configuration JSON Textarea === */}
                        <div className="space-y-2">
                            <Label htmlFor="config-json">Configuration JSON <span className="text-red-500">*</span></Label>
                            <Textarea
                                id="config-json"
                                placeholder='Paste your JSON config here...'
                                value={configJson}
                                onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setConfigJson(e.target.value)}
                                rows={8} // Adjust height as needed
                                required
                                className="font-mono text-sm" // Use monospace font for JSON
                                disabled={isSubmitting || taskState.isActive}
                            />
                            <p className="text-xs text-muted-foreground">
                                Must be valid JSON. **Include the `"model_class"` key matching the class name in your Model Definition file.**
                            </p>
                        </div>
                        {/* === END Added Textarea === */}


                        {/* GA Parameters (Consider if these are redundant if set in JSON) */}
                        <div className="grid grid-cols-2 gap-4">
                             <div>
                                <Label htmlFor="generations">Generations (From JSON)</Label>
                                <Input id="generations" type="number" min="1" value={generations} onChange={(e) => setGenerations(Math.max(1, parseInt(e.target.value, 10) || 1))} disabled={isSubmitting || taskState.isActive} />
                            </div>
                             <div>
                                <Label htmlFor="pop-size">Population Size (From JSON)</Label>
                                <Input id="pop-size" type="number" min="2" value={popSize} onChange={(e) => setPopSize(Math.max(2, parseInt(e.target.value, 10) || 2))} disabled={isSubmitting || taskState.isActive} />
                            </div>
                            {/* TODO: Add inputs for mutation rate/strength if needed, or remove if only set via JSON */}
                        </div>

                        {/* Submit Button */}
                        <Button type="submit" className="w-full" disabled={isSubmitting || taskState.isActive || (evalChoice === 'custom' && !taskEvalFile) || !modelDefFile}>
                            {isSubmitting ? "Submitting..." : taskState.isActive ? "Task Running..." : "Start Evolution"}
                        </Button>
                    </form>
                </CardContent>
            </Card>

            {/* --- Status & Plot Card (Remains the same) --- */}
            <Card>
                 <CardHeader>
                    <CardTitle>Task Status & Results</CardTitle>
                    <CardDescription>Monitor the evolution progress in real-time.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                   <AnimatePresence>
                    {taskState.taskId ? ( // Render only if a task has been started
                         <motion.div
                            key={taskState.taskId} // Use taskId as key for re-animation on new task
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="space-y-2"
                         >
                            <p>Task ID: <span className="font-mono text-sm bg-muted px-1 rounded">{taskState.taskId}</span></p>
                            <p>Status: <span className={`font-semibold ${taskState.status === 'SUCCESS' ? 'text-green-600' : taskState.status === 'FAILURE' ? 'text-red-600' : ''}`}>{taskState.status || 'N/A'}</span></p>
                            {/* Progress Bar */}
                            {(taskState.status === 'PROGRESS' || taskState.status === 'STARTED') && taskState.progress !== null && (
                                <div className="pt-1">
                                    <Progress value={taskState.progress * 100} className="w-full" />
                                    <p className="text-sm text-muted-foreground pt-1">{Math.round(taskState.progress * 100)}% complete</p>
                                </div>
                            )}
                             {taskState.message && <p className="text-sm text-muted-foreground">{taskState.message}</p>}

                             {/* --- Display Task Error --- */}
                             {taskState.error && (
                                <Alert variant="destructive" className="mt-2">
                                    <AlertCircle className="h-4 w-4" /> {/* Icon added */}
                                    <AlertTitle>Task Error</AlertTitle>
                                    <AlertDescription>
                                        {taskState.error}
                                    </AlertDescription>
                                </Alert>
                             )}

                             {/* --- Download Button (Uses Corrected Link) --- */}
                             {taskState.status === 'SUCCESS' && downloadLink && (
                                <Button variant="outline" size="sm" asChild className="mt-2">
                                   <a href={downloadLink} download>Download Final Model (.pth)</a>
                                </Button>
                             )}
                             {/* Cancel Button (optional - requires backend endpoint) */}
                             {/* {taskState.isActive && <Button variant="destructive" size="sm" onClick={handleCancel}>Cancel Task</Button>} */}
                         </motion.div>
                    ) : (
                        // Placeholder when no task is active/submitted yet
                         <p className="text-muted-foreground">Submit a task to see status and results.</p>
                    )}
                    </AnimatePresence>

                     {/* Real-time Plot Area */}
                     <div className="mt-4 h-64 border rounded bg-muted/20 flex items-center justify-center">
                         { (taskState.isActive || taskState.status === 'SUCCESS') && fitnessHistory.length > 0 ? (
                             <RealTimePlot data={fitnessHistory} /> // Pass fitness history data
                         ) : (
                            <p className="text-muted-foreground">{taskState.taskId ? "Plot will appear here..." : "Submit task for plot"}</p>
                         )}
                     </div>
                </CardContent>
            </Card>
        </div>
    );
}
