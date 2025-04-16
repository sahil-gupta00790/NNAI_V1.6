// components/evolver-section.tsx
'use client';
import React, { useState, useRef, ChangeEvent, FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, Loader2 } from "lucide-react"; // Import Loader2 for loading spinner
import { startEvolutionTask, analyzeGaResults } from '@/lib/api'; // Import analyzeGaResults
import { useTaskPolling } from '@/lib/hooks/useTaskPolling';
import RealTimePlot from './real-time-plot';
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from 'react-markdown'; // Import ReactMarkdown

// Default JSON config (Unchanged from previous correct version)
const defaultJsonConfig = JSON.stringify(
  {
    "model_class": "MyCNN",
    "generations": 20,
    "population_size": 30,
    "selection_strategy": "tournament",
    "crossover_operator": "one_point",
    "mutation_operator": "gaussian",
    "elitism_count": 1,
    "mutation_rate": 0.15,
    "mutation_strength": 0.05,
    "eval_config": {
      "batch_size": 128
    }
  }, null, 2
);

interface SubmittedConfig {
    generations?: number;
    population_size?: number;
    mutation_rate?: number;
    mutation_strength?: number;
    // Add other expected config fields used in handleAnalyzeClick if needed
  }

export default function EvolverSection() {
    // --- Existing State ---
    const [modelDefFile, setModelDefFile] = useState<File | null>(null);
    const [taskEvalFile, setTaskEvalFile] = useState<File | null>(null);
    const [weightsFile, setWeightsFile] = useState<File | null>(null);
    const [configJson, setConfigJson] = useState<string>(defaultJsonConfig);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [evalChoice, setEvalChoice] = useState<'standard' | 'custom'>('standard');
    // --- End Existing State ---

    // --- NEW State for Gemini Analysis ---
    const [analysisResult, setAnalysisResult] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
    // --- End NEW State ---

    // Refs (Unchanged)
    const modelDefRef = useRef<HTMLInputElement>(null);
    const taskEvalRef = useRef<HTMLInputElement>(null);
    const weightsRef = useRef<HTMLInputElement>(null);

    // Hook (Unchanged)
    const { taskState, startTask, resetTaskState } = useTaskPolling('evolver');

    // Event Handlers (Unchanged)
    const handleEvalChoiceChange = (value: 'standard' | 'custom') => {
        setEvalChoice(value);
        if (value === 'standard' && taskEvalRef.current) {
            taskEvalRef.current.value = ""; setTaskEvalFile(null);
        }
    };

    // handleSubmit (Unchanged)
    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        // Validations
        if (!modelDefFile) { toast.error("Model Definition file is required."); return; }
        if (evalChoice === 'custom' && !taskEvalFile) { toast.error("Custom Evaluation Script file is required."); return; }
        if (taskState.isActive) { toast.warning("A task is already running."); return; }
        let parsedConfig;
        try {
            parsedConfig = JSON.parse(configJson);
            // Basic validation - keep as is
            if (!parsedConfig.model_class || typeof parsedConfig.model_class !== 'string' || parsedConfig.model_class.trim() === "") { throw new Error("'model_class' key (string) is missing or invalid."); }
            if (typeof parsedConfig.generations !== 'number' || parsedConfig.generations < 1) { throw new Error("'generations' must be a positive number."); }
            if (typeof parsedConfig.population_size !== 'number' || parsedConfig.population_size < 2) { throw new Error("'population_size' must be at least 2."); }
        } catch (jsonError: any) { toast.error(`Invalid Configuration JSON: ${jsonError.message}`); return; }

        console.log("Frontend: Sending configJson string:", configJson);

        setIsSubmitting(true);
        resetTaskState(); // Reset task state before starting new one
        setAnalysisResult(null); // Also clear previous analysis on new run
        setIsAnalyzing(false); // Ensure analysis state is reset
        toast("Submitting evolution task...");

        const formData = new FormData();
        formData.append('model_definition', modelDefFile);
        formData.append('use_standard_eval', String(evalChoice === 'standard'));
        if (evalChoice === 'custom' && taskEvalFile) formData.append('task_evaluation', taskEvalFile);
        if (weightsFile) formData.append('initial_weights', weightsFile);
        formData.append('config_json', configJson);

        try {
            const response = await startEvolutionTask(formData);
            startTask(response.task_id);
            toast.success(`Task ${response.task_id} started.`);
             if(modelDefRef.current) modelDefRef.current.value = "";
             if(taskEvalRef.current) taskEvalRef.current.value = "";
             if(weightsRef.current) weightsRef.current.value = "";
             setModelDefFile(null); setTaskEvalFile(null); setWeightsFile(null);
        } catch (error: any) {
            console.error("Error starting evolution task:", error);
            toast.error(`Failed to start task: ${error.message || 'Unknown error'}`);
        } finally {
            setIsSubmitting(false);
        }
    };
    // --- End handleSubmit ---

    const handleAnalyzeClick = async () => {
        // Ensure task is successful and result data (especially histories) is available
        if (taskState.status !== 'SUCCESS' || !taskState.result || !Array.isArray(taskState.result.fitness_history)) {
            toast.error("Task not successfully completed or result data is missing/invalid for analysis.");
            return;
        }

        setIsAnalyzing(true);
        setAnalysisResult(null); // Clear previous analysis

        try {
            // --- Parse submitted config safely ---
            let submittedConfig: SubmittedConfig = {}; // Use interface or keep as {}
            try {
                // Attempt parsing, use type assertion if interface exists, or keep as any
                submittedConfig = JSON.parse(configJson) as SubmittedConfig;
            } catch {
                toast.warning("Could not parse current JSON config for analysis context, using defaults.");
                // submittedConfig remains {}
            }

            // --- Prepare data payload with type safety ---
            const analysisPayload = {
                // Histories: Convert undefined from taskState to null for API call
                fitness_history: taskState.result.fitness_history, // Already checked this is an array
                avg_fitness_history: taskState.result.avg_fitness_history ?? null,
                diversity_history: taskState.result.diversity_history ?? null,

                // Config Context: Use ?. optional chaining and ?? nullish coalescing for defaults
                generations: submittedConfig?.generations ?? taskState.result.fitness_history.length, // Default to actual generations run
                population_size: submittedConfig?.population_size ?? 0, // Default to 0 or another sensible value if not parsed
                mutation_rate: submittedConfig?.mutation_rate, // Will be undefined if not present, which matches API expectation
                mutation_strength: submittedConfig?.mutation_strength // Will be undefined if not present
            };

            // Minimal validation before sending
            if (analysisPayload.generations <= 0) {
                 throw new Error("Invalid generation count for analysis.");
            }

            toast.info("Requesting analysis from Gemini AI...");
            const response = await analyzeGaResults(analysisPayload); // Pass the correctly typed payload
            setAnalysisResult(response.analysis_text);
            toast.success("Analysis received!");

        } catch (error: any) {
            console.error("Error fetching analysis:", error);
            toast.error(`Failed to generate analysis: ${error.message || 'Unknown error'}`);
        } finally {
            setIsAnalyzing(false);
        }
    };
    // --- End handleAnalyzeClick ---

    // Plot Data Prep (Unchanged)
    const plotData = {
        maxFitness: Array.isArray(taskState.fitnessHistory) ? taskState.fitnessHistory : [],
        avgFitness: Array.isArray(taskState.avgFitnessHistory) ? taskState.avgFitnessHistory : [],
        diversity: Array.isArray(taskState.diversityHistory) ? taskState.diversityHistory : []
    };
    const hasPlotData = plotData.maxFitness.length > 0 || plotData.avgFitness.length > 0 || plotData.diversity.length > 0;

    // Download Link (Unchanged)
    const downloadLink = taskState.status === 'SUCCESS' && taskState.result?.final_model_path && taskState.taskId
        ? `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/evolver/results/${taskState.taskId}/download`
        : undefined;

    // --- Component Return ---
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Form Card (Unchanged) */}
            <Card>
                {/* ... CardHeader ... */}
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {/* ... File Inputs, Radio Group, Config JSON Textarea, Submit Button ... */}
                         {/* File Inputs */}
                         <div>
                            <Label htmlFor="model-def">Model Definition (.py) <span className="text-red-500">*</span></Label>
                            <Input ref={modelDefRef} id="model-def" type="file" accept=".py" required onChange={(e: ChangeEvent<HTMLInputElement>) => setModelDefFile(e.target.files?.[0] ?? null)} disabled={isSubmitting || taskState.isActive} />
                        </div>
                        <div>
                            <Label>Evaluation Method <span className="text-red-500">*</span></Label>
                            <RadioGroup value={evalChoice} onValueChange={handleEvalChoiceChange} className="flex space-x-4 mt-1" disabled={isSubmitting || taskState.isActive}>
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
                        {/* Keep inner AnimatePresence */}
                        <AnimatePresence>
                            {evalChoice === 'custom' && (
                                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.3 }} className="mt-2">
                                    <Label htmlFor="task-eval">Custom Evaluation Script (.py) <span className="text-red-500">*</span></Label>
                                    <Input ref={taskEvalRef} id="task-eval" type="file" accept=".py" required={evalChoice === 'custom'} onChange={(e: ChangeEvent<HTMLInputElement>) => setTaskEvalFile(e.target.files?.[0] ?? null)} disabled={isSubmitting || taskState.isActive} />
                                </motion.div>
                             )}
                         </AnimatePresence>
                        <div>
                            <Label htmlFor="init-weights">Initial Weights (.pth, Optional)</Label>
                            <Input ref={weightsRef} id="init-weights" type="file" accept=".pth,.pt" onChange={(e: ChangeEvent<HTMLInputElement>) => setWeightsFile(e.target.files?.[0] ?? null)} disabled={isSubmitting || taskState.isActive} />
                        </div>
                        {/* Config JSON */}
                        <div className="space-y-2">
                            <Label htmlFor="config-json">Configuration JSON <span className="text-red-500">*</span></Label>
                            <Textarea
                                id="config-json"
                                placeholder='Edit JSON config for GA parameters...'
                                value={configJson}
                                onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setConfigJson(e.target.value)}
                                rows={12}
                                required
                                className="font-mono text-sm"
                                disabled={isSubmitting || taskState.isActive}
                            />
                            <p className="text-xs text-muted-foreground">Must be valid JSON. Edit parameters like `generations`, `population_size`, `selection_strategy`, `mutation_rate`, etc. Ensure `model_class` matches your uploaded file.</p>
                        </div>
                        {/* Submit Button */}
                        <Button type="submit" className="w-full" disabled={isSubmitting || taskState.isActive || !modelDefFile || (evalChoice === 'custom' && !taskEvalFile)}>
                            {isSubmitting ? "Submitting..." : taskState.isActive ? "Task Running..." : "Start Evolution"}
                        </Button>
                    </form>
                </CardContent>
            </Card>

            {/* Status & Plot Card */}
            <Card>
                 <CardHeader>
                    <CardTitle>Task Status & Results</CardTitle>
                    <CardDescription>Monitor the evolution progress in real-time.</CardDescription>
                 </CardHeader>
                <CardContent className="space-y-4">
                    {/* --- Status Display Block --- */}
                    {taskState.taskId ? (
                         <div key={taskState.taskId} className="space-y-2" >
                            {/* Status Info */}
                            <p>Task ID: <span className="font-mono text-sm bg-muted px-1 rounded">{taskState.taskId}</span></p>
                            <p>Status: <span className={`font-semibold ${taskState.status === 'SUCCESS' ? 'text-green-600' : taskState.status === 'FAILURE' ? 'text-red-600' : ''}`}>{taskState.status || 'N/A'}</span></p>
                            {(taskState.status === 'PROGRESS' || taskState.status === 'STARTED') && typeof taskState.progress === 'number' && (
                                <div className="pt-1">
                                    <Progress value={taskState.progress * 100} className="w-full" />
                                    <p className="text-sm text-muted-foreground pt-1">{Math.round(taskState.progress * 100)}% complete</p>
                                </div>
                            )}
                            {taskState.message && <p className="text-sm text-muted-foreground">{taskState.message}</p>}
                            {taskState.error && (
                                <Alert variant="destructive" className="mt-2"> <AlertCircle className="h-4 w-4" /> <AlertTitle>Task Error</AlertTitle> <AlertDescription>{taskState.error}</AlertDescription> </Alert>
                            )}
                            {/* --- Download Button --- */}
                            {taskState.status === 'SUCCESS' && downloadLink !== undefined && (
                                <Button variant="outline" size="sm" asChild className="mt-2">
                                    <a href={downloadLink} download>Download Final Model (.pth)</a>
                                </Button>
                            )}

                            {/* --- NEW: Gemini Analysis Section --- */}
                            {taskState.status === 'SUCCESS' && taskState.result?.fitness_history && ( // Render only if successful and history exists
                                <div className="mt-4 pt-4 border-t"> {/* Add separator */}
                                    <Button
                                        onClick={handleAnalyzeClick}
                                        disabled={isAnalyzing || !taskState.result?.fitness_history} // Disable if no history
                                        variant="secondary"
                                    >
                                        {isAnalyzing ? (
                                            <> <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing... </>
                                        ) : ( "Analyze Results with Gemini" )}
                                    </Button>

                                    {/* Loading Indicator */}
                                    {isAnalyzing && (
                                        <p className="text-sm text-muted-foreground mt-2"> Contacting Gemini AI, this may take a moment... </p>
                                    )}

                                    {/* Analysis Result Display */}
                                    {analysisResult && !isAnalyzing && ( // Show only when not loading and result exists
                                        <Card className="mt-4 bg-muted/50"> {/* Slightly different background */}
                                             <CardHeader className="pb-2 pt-4">
                                                 <CardTitle className="text-lg">Gemini Analysis</CardTitle>
                                             </CardHeader>
                                            <CardContent className="p-4 prose dark:prose-invert prose-sm max-w-none">
                                                {/* Render analysis text using ReactMarkdown */}
                                                <ReactMarkdown
                                                    components={{ // Optional: Customize rendering if needed
                                                        // Example: Make links open in new tab
                                                        // a: ({node, ...props}) => <a target="_blank" rel="noopener noreferrer" {...props} />
                                                    }}
                                                >
                                                    {analysisResult}
                                                </ReactMarkdown>
                                            </CardContent>
                                        </Card>
                                    )}
                                </div>
                            )}
                            {/* --- End Gemini Analysis Section --- */}

                         </div>
                    ) : (
                        <p className="text-muted-foreground">Submit a task to see status and results.</p>
                    )}
                     {/* --- End Status Block --- */}

                     {/* Plot Area (Unchanged) */}
                     <div className="mt-4 h-72 border rounded bg-muted/20 flex items-center justify-center">
                         { hasPlotData ? (
                             <RealTimePlot
                                maxFitnessData={plotData.maxFitness}
                                avgFitnessData={plotData.avgFitness}
                                diversityData={plotData.diversity}
                             />
                         ) : ( <p className="text-muted-foreground">{taskState.taskId ? "Plot will appear here..." : "Submit task for plot"}</p> )}
                     </div>
                </CardContent>
            </Card>
        </div>
    );
}
