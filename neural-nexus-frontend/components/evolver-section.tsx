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
import { AlertCircle } from "lucide-react";
import { startEvolutionTask } from '@/lib/api';
import { useTaskPolling } from '@/lib/hooks/useTaskPolling';
import RealTimePlot from './real-time-plot';
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion"; // Keep imports even if unused temporarily

// --- CORRECTED Default JSON config ---
const defaultJsonConfig = JSON.stringify(
  {
    model_class: "MyCNN",
    generations: 20,
    population_size: 30,
    selection_strategy: "tournament",
    crossover_operator: "one_point",
    mutation_operator: "gaussian",
    elitism_count: 1,
    mutation_rate: 0.15,
    mutation_strength: 0.05,
    // --- CORRECTED eval_config key ---
    "eval_config": {
      "batch_size": 128 // Use underscore, ensure quotes are correct
    }
    // --- END CORRECTION ---
  }, null, 2
);

export default function EvolverSection() {
    // State (Unchanged)
    const [modelDefFile, setModelDefFile] = useState<File | null>(null);
    const [taskEvalFile, setTaskEvalFile] = useState<File | null>(null);
    const [weightsFile, setWeightsFile] = useState<File | null>(null);
    const [configJson, setConfigJson] = useState<string>(defaultJsonConfig);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [evalChoice, setEvalChoice] = useState<'standard' | 'custom'>('standard');

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

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => { // Explicit type for 'e'
        e.preventDefault();
        // Validations (Unchanged)
        if (!modelDefFile) { toast.error("Model Definition file is required."); return; }
        if (evalChoice === 'custom' && !taskEvalFile) { toast.error("Custom Evaluation Script file is required."); return; }
        if (taskState.isActive) { toast.warning("A task is already running."); return; }
        let parsedConfig;
        try {
            parsedConfig = JSON.parse(configJson);
            if (!parsedConfig.model_class || typeof parsedConfig.model_class !== 'string' || parsedConfig.model_class.trim() === "") { throw new Error("'model_class' key (string) is missing or invalid."); }
            if (typeof parsedConfig.generations !== 'number' || parsedConfig.generations < 1) { throw new Error("'generations' must be a positive number."); }
            if (typeof parsedConfig.population_size !== 'number' || parsedConfig.population_size < 2) { throw new Error("'population_size' must be at least 2."); }
        } catch (jsonError: any) { toast.error(`Invalid Configuration JSON: ${jsonError.message}`); return; }

        console.log("Frontend: Sending configJson string:", configJson);

        setIsSubmitting(true);
        resetTaskState();
        toast("Submitting evolution task...");

        // FormData (Unchanged)
        const formData = new FormData();
        formData.append('model_definition', modelDefFile);
        formData.append('use_standard_eval', String(evalChoice === 'standard'));
        if (evalChoice === 'custom' && taskEvalFile) formData.append('task_evaluation', taskEvalFile);
        if (weightsFile) formData.append('initial_weights', weightsFile);
        formData.append('config_json', configJson);

        // API Call (Unchanged)
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

    // Plot Data Prep (Unchanged)
    const plotData = {
        maxFitness: Array.isArray(taskState.fitnessHistory) ? taskState.fitnessHistory : [],
        avgFitness: Array.isArray(taskState.avgFitnessHistory) ? taskState.avgFitnessHistory : [],
        diversity: Array.isArray(taskState.diversityHistory) ? taskState.diversityHistory : []
    };
    const hasPlotData = plotData.maxFitness.length > 0 || plotData.avgFitness.length > 0 || plotData.diversity.length > 0;

    // Download Link (Handle null for href)
    const downloadLink = taskState.status === 'SUCCESS' && taskState.result?.final_model_path && taskState.taskId
        ? `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/evolver/results/${taskState.taskId}/download`
        : undefined; // Use undefined for missing href, null is invalid

    // --- Component Return ---
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Form Card */}
            <Card>
                <CardHeader>
                    <CardTitle>Configure Evolution</CardTitle>
                    <CardDescription>Upload files and configure GA parameters via JSON.</CardDescription>
                </CardHeader>
                <CardContent>
                    {/* --- CORRECTED JSX STRUCTURE --- */}
                    <form onSubmit={handleSubmit} className="space-y-4">
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
                        {/* Keep inner AnimatePresence, ensure it wraps ONLY the conditional element */}
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
                     {/* --- END CORRECTED JSX --- */}
                </CardContent>
            </Card>

            {/* Status & Plot Card */}
            <Card>
                 <CardHeader>
                    <CardTitle>Task Status & Results</CardTitle>
                    <CardDescription>Monitor the evolution progress in real-time.</CardDescription>
                 </CardHeader>
                <CardContent className="space-y-4">
                   {/* Still keep outer AnimatePresence/motion.div removed for testing */}
                    {taskState.taskId ? (
                         <div key={taskState.taskId} className="space-y-2" > {/* Use simple div */}
                            {/* Status Display Content */}
                            <p>Task ID: <span className="font-mono text-sm bg-muted px-1 rounded">{taskState.taskId}</span></p>
                            <p>Status: <span className={`font-semibold ${taskState.status === 'SUCCESS' ? 'text-green-600' : taskState.status === 'FAILURE' ? 'text-red-600' : ''}`}>{taskState.status || 'N/A'}</span></p>
                            {/* Null check for progress */}
                            {(taskState.status === 'PROGRESS' || taskState.status === 'STARTED') && typeof taskState.progress === 'number' && (
                                <div className="pt-1">
                                    <Progress value={taskState.progress * 100} className="w-full" />
                                    <p className="text-sm text-muted-foreground pt-1">{Math.round(taskState.progress * 100)}% complete</p>
                                </div>
                            )}
                            {taskState.message && <p className="text-sm text-muted-foreground">{taskState.message}</p>}
                            {taskState.error && (
                                <Alert variant="destructive" className="mt-2">
                                    <AlertCircle className="h-4 w-4" />
                                    <AlertTitle>Task Error</AlertTitle>
                                    <AlertDescription>{taskState.error}</AlertDescription>
                                </Alert>
                            )}
                            {/* Download Button - href needs undefined check */}
                            {taskState.status === 'SUCCESS' && downloadLink && (
                                <Button variant="outline" size="sm" asChild className="mt-2">
                                    <a href={downloadLink} download>Download Final Model (.pth)</a>
                                </Button>
                            )}
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
                </CardContent> {/* Ensure CardContent closes */}
            </Card> {/* Ensure Card closes */}
        </div> // Ensure outer div closes
    );
} // Ensure export default closes
