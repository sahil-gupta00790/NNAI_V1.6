// components/evolver-section.tsx
'use client';
import React, { useState, useRef, ChangeEvent, FormEvent, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, Loader2, HelpCircle } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Switch } from '@/components/ui/switch';
import { startEvolutionTask, analyzeGaResults } from '@/lib/api';
import { useTaskPolling } from '@/lib/hooks/useTaskPolling';
import RealTimePlot from './real-time-plot';
import { toast } from "sonner";
import { motion } from "framer-motion";
import ReactMarkdown from 'react-markdown';

// --- Interface for schema items ---
interface GaParameterSchemaItem {
    name: string;
    label: string;
    type: 'number' | 'float' | 'select' | 'slider' | 'boolean';
    defaultValue: any;
    min?: number;
    max?: number;
    step?: number;
    description: string;
    condition?: (formData: Record<string, any>) => boolean;
    options?: { value: string; label: string }[];
}

// --- Define the schema for GA parameters ---
const gaParameterSchema: GaParameterSchemaItem[] = [
  // --- Standard Params ---
  { name: 'generations', label: 'Generations', type: 'number', defaultValue: 20, min: 1, max: 500, step: 1, description: 'Number of generations the evolution will run.' },
  { name: 'population_size', label: 'Population Size', type: 'number', defaultValue: 30, min: 2, max: 500, step: 1, description: 'Number of individuals in each generation.' },
  { name: 'selection_strategy', label: 'Selection Strategy', type: 'select', defaultValue: 'tournament', options: [ { value: 'tournament', label: 'Tournament' }, { value: 'roulette', label: 'Roulette Wheel' } ], description: 'Method for selecting parents for reproduction.'},
  { name: 'tournament_size', label: 'Tournament Size', type: 'number', defaultValue: 3, min: 2, max: 20, step: 1, condition: (formData) => formData.selection_strategy === 'tournament', description: 'Number of individuals competing in each tournament selection.' },
  { name: 'crossover_operator', label: 'Crossover Operator', type: 'select', defaultValue: 'one_point', options: [ { value: 'one_point', label: 'One Point' }, { value: 'uniform', label: 'Uniform' }, { value: 'average', label: 'Average' } ], description: 'Method for combining parent weights to create offspring.' },
  { name: 'uniform_crossover_prob', label: 'Uniform Crossover Prob', type: 'slider', defaultValue: 0.5, min: 0, max: 1, step: 0.01, condition: (formData) => formData.crossover_operator === 'uniform', description: 'Probability of swapping genes between parents in uniform crossover.' },
  { name: 'mutation_operator', label: 'Mutation Operator', type: 'select', defaultValue: 'gaussian', options: [ { value: 'gaussian', label: 'Gaussian Noise' }, { value: 'uniform_random', label: 'Uniform Random Replacement' } ], description: 'Method for introducing random changes to offspring weights.' },

  // --- Dynamic Mutation Rate Control ---
  { name: 'use_dynamic_mutation_rate', label: 'Dynamic Mut. Rate', type: 'boolean', defaultValue: false, description: 'Enable adaptive mutation rate strategies for weights. Disables the fixed rate slider below.' },
  {
    name: 'dynamic_mutation_heuristic',
    label: 'Heuristic Strategy',
    type: 'select',
    defaultValue: 'time_decay',
    options: [
      { value: 'time_decay', label: 'Time-Based Decay' },
      { value: 'fitness_based', label: 'Fitness-Based (Stagnation)' },
      { value: 'diversity_based', label: 'Diversity-Based' },
    ],
    condition: (formData) => !!formData.use_dynamic_mutation_rate,
    description: 'Method for adapting the weight mutation rate during the run.'
  },

  // --- Original Fixed Mutation Rate (Conditional) ---
  {
    name: 'mutation_rate',
    label: 'Mutation Rate',
    type: 'slider',
    defaultValue: 0.15,
    min: 0, max: 1, step: 0.01,
    description: 'Probability of each weight being mutated (used only when Dynamic Rate is OFF).',
    condition: (formData) => !formData.use_dynamic_mutation_rate
  },

  // --- Time-Based Decay Params (Conditional) ---
  { name: 'initial_mutation_rate', label: 'Initial Rate', type: 'slider', defaultValue: 0.2, min: 0, max: 1, step: 0.01, condition: (formData) => !!formData.use_dynamic_mutation_rate && formData.dynamic_mutation_heuristic === 'time_decay', description: 'Starting weight mutation rate for time-based decay.' },
  { name: 'final_mutation_rate', label: 'Final Rate', type: 'slider', defaultValue: 0.01, min: 0, max: 1, step: 0.01, condition: (formData) => !!formData.use_dynamic_mutation_rate && formData.dynamic_mutation_heuristic === 'time_decay', description: 'Target weight mutation rate at the final generation.' },

  // --- Fitness-Based Params (Conditional) ---
  { name: 'normal_fitness_mutation_rate', label: 'Normal Rate', type: 'slider', defaultValue: 0.05, min: 0, max: 1, step: 0.01, condition: (formData) => !!formData.use_dynamic_mutation_rate && formData.dynamic_mutation_heuristic === 'fitness_based', description: 'Weight mutation rate used when fitness is improving sufficiently.' },
  { name: 'stagnation_mutation_rate', label: 'Increased Rate', type: 'slider', defaultValue: 0.25, min: 0, max: 1, step: 0.01, condition: (formData) => !!formData.use_dynamic_mutation_rate && formData.dynamic_mutation_heuristic === 'fitness_based', description: 'Weight mutation rate used when average fitness improvement is below the threshold.' },
  { name: 'stagnation_threshold', label: 'Stagnation Threshold', type: 'float', defaultValue: 0.001, min: 0, step: 0.0001, condition: (formData) => !!formData.use_dynamic_mutation_rate && formData.dynamic_mutation_heuristic === 'fitness_based', description: 'Minimum average fitness improvement required to use the "Normal Rate".' },

  // --- Diversity-Based Params (Conditional) ---
  { name: 'base_mutation_rate', label: 'Base Rate', type: 'slider', defaultValue: 0.1, min: 0, max: 1, step: 0.01, condition: (formData) => !!formData.use_dynamic_mutation_rate && formData.dynamic_mutation_heuristic === 'diversity_based', description: 'Base weight mutation rate used when diversity is adequate.' },
  { name: 'diversity_threshold_low', label: 'Low Div. Threshold', type: 'float', defaultValue: 0.1, min: 0, step: 0.01, condition: (formData) => !!formData.use_dynamic_mutation_rate && formData.dynamic_mutation_heuristic === 'diversity_based', description: 'Weight diversity value below which mutation rate increases.' },
  { name: 'mutation_rate_increase_factor', label: 'Rate Increase Factor', type: 'float', defaultValue: 1.5, min: 1, step: 0.1, condition: (formData) => !!formData.use_dynamic_mutation_rate && formData.dynamic_mutation_heuristic === 'diversity_based', description: 'Factor to multiply base rate by when weight diversity is low.' },

  // --- Mutation Strength ---
  { name: 'mutation_strength', label: 'Mutation Strength', type: 'float', defaultValue: 0.05, min: 0, step: 0.001, description: 'Magnitude of weight mutation (e.g., std dev for Gaussian). Applies always.' },

  // --- Other Params ---
  { name: 'elitism_count', label: 'Elitism Count', type: 'number', defaultValue: 1, min: 0, max: 10, step: 1, description: 'Number of best individuals carried directly to the next generation.' },
  { name: 'eval_batch_size', label: 'Eval Batch Size', type: 'number', defaultValue: 128, min: 1, max: 1024, step: 1, description: 'Batch size used during model evaluation (passed in eval_config).'}
];
// --- End Schema ---

// --- Helper to initialize state from schema ---
function getInitialFormData() {
  const initialData: Record<string, any> = {};
  gaParameterSchema.forEach((param: GaParameterSchemaItem) => {
    initialData[param.name] = param.defaultValue ?? (param.type === 'boolean' ? false : null);
  });
  initialData['model_class'] = 'MyCNN';
  initialData['model_args'] = [];
  initialData['model_kwargs'] = {};
  initialData['evolvable_hyperparams'] = {};
  initialData['eval_config'] = { batch_size: initialData['eval_batch_size'] };
  return initialData;
}
// --- End Helper ---

export default function EvolverSection() {
    // --- State ---
    const [modelDefFile, setModelDefFile] = useState<File | null>(null);
    const [taskEvalFile, setTaskEvalFile] = useState<File | null>(null);
    const [weightsFile, setWeightsFile] = useState<File | null>(null);
    const [formData, setFormData] = useState<Record<string, any>>(getInitialFormData());
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [evalChoice, setEvalChoice] = useState<'standard' | 'custom'>('standard');
    const [analysisResult, setAnalysisResult] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
    // --- End State ---

    // Refs (Unchanged)
    const modelDefRef = useRef<HTMLInputElement>(null);
    const taskEvalRef = useRef<HTMLInputElement>(null);
    const weightsRef = useRef<HTMLInputElement>(null);

    // Hook (Unchanged)
    const { taskState, startTask, resetTaskState } = useTaskPolling('evolver');

    // --- General Form Change Handler ---
    const handleFormChange = (name: string, value: any) => {
        let processedValue = value;
        const schemaItem = gaParameterSchema.find(p => p.name === name);

        if (schemaItem) {
             if (schemaItem.type === 'number' || schemaItem.type === 'float' || schemaItem.type === 'slider') {
                const numValue = parseFloat(value);
                processedValue = (value === '' || isNaN(numValue)) ? null : numValue;
            } else if (schemaItem.type === 'boolean') {
                processedValue = !!value;
            }
        }

        if (name === 'eval_batch_size') {
             setFormData(prev => ({
                 ...prev,
                 [name]: value === '' ? '' : processedValue,
                 eval_config: { ...(prev.eval_config || {}), batch_size: typeof processedValue === 'number' ? processedValue : undefined }
             }));
        } else {
             setFormData(prev => ({ ...prev, [name]: processedValue }));
        }
    };
    // --- End Change Handler ---

    // Event Handlers (Unchanged)
    const handleEvalChoiceChange = (value: 'standard' | 'custom') => { /* ... */
        setEvalChoice(value);
        if (value === 'standard' && taskEvalRef.current) { taskEvalRef.current.value = ""; setTaskEvalFile(null); }
    };
    const handleModelDefFileChange = (e: ChangeEvent<HTMLInputElement>) => { /* ... */ setModelDefFile(e.target.files?.[0] ?? null); };
    const handleTaskEvalFileChange = (e: ChangeEvent<HTMLInputElement>) => { /* ... */ setTaskEvalFile(e.target.files?.[0] ?? null); };
    const handleWeightsFileChange = (e: ChangeEvent<HTMLInputElement>) => { /* ... */ setWeightsFile(e.target.files?.[0] ?? null); };

    // handleSubmit (Unchanged from previous)
    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (!modelDefFile) { toast.error("Model Definition file is required."); return; }
        if (evalChoice === 'custom' && !taskEvalFile) { toast.error("Custom Evaluation Script file is required."); return; }
        if (taskState.isActive) { toast.warning("A task is already running."); return; }

        const finalFormDataObject = { ...formData };

        if (typeof finalFormDataObject['eval_batch_size'] === 'number') { finalFormDataObject['eval_config'] = { ...(finalFormDataObject['eval_config'] || {}), batch_size: finalFormDataObject['eval_batch_size'] }; }
        else if (finalFormDataObject['eval_config']) { delete finalFormDataObject['eval_config'].batch_size; }
        delete finalFormDataObject['eval_batch_size'];

        // --- Validation ---
        const requiredNumericFields = ['generations', 'population_size', 'elitism_count'];
        if (!finalFormDataObject.use_dynamic_mutation_rate) { if (finalFormDataObject['mutation_rate'] === null || typeof finalFormDataObject['mutation_rate'] !== 'number') { toast.error("'Mutation Rate (Fixed)' must be valid."); return; } }
        else {
            const heuristic = finalFormDataObject.dynamic_mutation_heuristic;
            if (heuristic === 'time_decay') { if (finalFormDataObject['initial_mutation_rate'] === null || finalFormDataObject['final_mutation_rate'] === null) { toast.error("Initial/Final Rate required for Time Decay."); return; } }
            else if (heuristic === 'fitness_based') { if (finalFormDataObject['normal_fitness_mutation_rate'] === null || finalFormDataObject['stagnation_mutation_rate'] === null || finalFormDataObject['stagnation_threshold'] === null) { toast.error("Normal/Increased Rate & Threshold required for Fitness-Based."); return; } }
            else if (heuristic === 'diversity_based') { if (finalFormDataObject['base_mutation_rate'] === null || finalFormDataObject['diversity_threshold_low'] === null || finalFormDataObject['mutation_rate_increase_factor'] === null) { toast.error("Base Rate, Threshold & Factor required for Diversity-Based."); return; } }
        }
        if (finalFormDataObject['mutation_strength'] === null || typeof finalFormDataObject['mutation_strength'] !== 'number' || finalFormDataObject['mutation_strength'] < 0) { toast.error("'Mutation Strength (Weights)' must be non-negative."); return; }
        for (const field of requiredNumericFields) {
            const schemaItem = gaParameterSchema.find(p => p.name === field);
            if (finalFormDataObject[field] === null || typeof finalFormDataObject[field] !== 'number' || finalFormDataObject[field] < (schemaItem?.min ?? 0)) { toast.error(`'${schemaItem?.label || field}' is invalid.`); return; }
        }
        if (!finalFormDataObject.model_class || !finalFormDataObject.model_class.trim()) { toast.error("Internal Error: Model class missing."); return; }
        // --- End Validation ---

        const configJsonString = JSON.stringify(finalFormDataObject, null, 2);
        console.log("Frontend: Sending final config JSON:", configJsonString);

        setIsSubmitting(true); resetTaskState(); setAnalysisResult(null); setIsAnalyzing(false);
        toast("Submitting evolution task...");

        const apiFormData = new FormData();
        apiFormData.append('model_definition', modelDefFile);
        apiFormData.append('use_standard_eval', String(evalChoice === 'standard'));
        if (evalChoice === 'custom' && taskEvalFile) apiFormData.append('task_evaluation', taskEvalFile);
        if (weightsFile) apiFormData.append('initial_weights', weightsFile);
        apiFormData.append('config_json', configJsonString);

        try {
            const response = await startEvolutionTask(apiFormData);
            startTask(response.task_id);
            toast.success(`Task ${response.task_id} started.`);
             if(modelDefRef.current) modelDefRef.current.value = "";
             if(taskEvalRef.current) taskEvalRef.current.value = "";
             if(weightsRef.current) weightsRef.current.value = "";
             setModelDefFile(null); setTaskEvalFile(null); setWeightsFile(null);
        } catch (error: any) {
            console.error("Error starting evolution task:", error);
            toast.error(`Failed to start task: ${error.message || 'Unknown error'}`);
            resetTaskState();
        } finally { setIsSubmitting(false); }
    };
    // --- End handleSubmit ---

    // handleAnalyzeClick (Unchanged)
    const handleAnalyzeClick = async () => {
        if (taskState.status !== 'SUCCESS' || !taskState.result || !Array.isArray(taskState.result.fitness_history)) { toast.error("Task not successfully completed or result data is missing/invalid for analysis."); return; }
        setIsAnalyzing(true); setAnalysisResult(null);
        try {
            const { fitness_history, avg_fitness_history, diversity_history } = taskState.result;
            let generations: number | undefined = typeof formData.generations === 'number' ? formData.generations : undefined;
            if (!generations || generations <= 0) { const historyLength = fitness_history?.length; if (historyLength && historyLength > 0) { generations = historyLength; } else { toast.error("Invalid generation count for analysis."); setIsAnalyzing(false); return; } }
            const population_size: number | undefined = typeof formData.population_size === 'number' ? formData.population_size : undefined;
            if (typeof population_size !== 'number') { toast.error("Invalid population size for analysis."); setIsAnalyzing(false); return; }
            const basePayload = { fitness_history, avg_fitness_history: avg_fitness_history ?? null, diversity_history: diversity_history ?? null, generations, population_size, mutation_strength: typeof formData.mutation_strength === 'number' ? formData.mutation_strength : undefined };
            let dynamicParams = {};
            if (formData.use_dynamic_mutation_rate) {
                dynamicParams = { use_dynamic_mutation_rate: true, dynamic_mutation_heuristic: formData.dynamic_mutation_heuristic, initial_mutation_rate: typeof formData.initial_mutation_rate === 'number' ? formData.initial_mutation_rate : undefined, final_mutation_rate: typeof formData.final_mutation_rate === 'number' ? formData.final_mutation_rate : undefined, normal_fitness_mutation_rate: typeof formData.normal_fitness_mutation_rate === 'number' ? formData.normal_fitness_mutation_rate : undefined, stagnation_mutation_rate: typeof formData.stagnation_mutation_rate === 'number' ? formData.stagnation_mutation_rate : undefined, stagnation_threshold: typeof formData.stagnation_threshold === 'number' ? formData.stagnation_threshold : undefined, base_mutation_rate: typeof formData.base_mutation_rate === 'number' ? formData.base_mutation_rate : undefined, diversity_threshold_low: typeof formData.diversity_threshold_low === 'number' ? formData.diversity_threshold_low : undefined, mutation_rate_increase_factor: typeof formData.mutation_rate_increase_factor === 'number' ? formData.mutation_rate_increase_factor : undefined };
            } else { dynamicParams = { use_dynamic_mutation_rate: false, mutation_rate: typeof formData.mutation_rate === 'number' ? formData.mutation_rate : undefined }; }
            const finalPayload = { ...basePayload, ...dynamicParams };
            toast.info("Requesting analysis from Gemini AI...");
            console.log("Sending to /analyze:", finalPayload);
            const response = await analyzeGaResults(finalPayload);
            setAnalysisResult(response.analysis_text); toast.success("Analysis received!");
        } catch (error: any) { console.error("Error fetching analysis:", error); toast.error(`Failed to generate analysis: ${error.message || 'Unknown error'}`); }
        finally { setIsAnalyzing(false); }
     };
    // --- End handleAnalyzeClick ---

    // Plot Data Prep, Download Link (Unchanged)
    const plotData = {
        maxFitness: Array.isArray(taskState.fitnessHistory) ? taskState.fitnessHistory : [],
        avgFitness: Array.isArray(taskState.avgFitnessHistory) ? taskState.avgFitnessHistory : [],
        diversity: Array.isArray(taskState.diversityHistory) ? taskState.diversityHistory : []
    };
    const hasPlotData = plotData.maxFitness.length > 0 || plotData.avgFitness.length > 0 || plotData.diversity.length > 0;
    const downloadLink = taskState.status === 'SUCCESS' && taskState.result?.final_model_path && taskState.taskId ? `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/evolver/results/${taskState.taskId}/download` : undefined;

    // --- Component Return ---
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Form Card */}
            <Card>
                <CardHeader> <CardTitle>Configure Evolution</CardTitle> <CardDescription>Set files and Genetic Algorithm parameters.</CardDescription> </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* File Inputs */}
                        <div> <Label htmlFor="model-def">Model Definition (.py) <span className="text-red-500">*</span></Label> <Input ref={modelDefRef} id="model-def" type="file" accept=".py" required onChange={handleModelDefFileChange} disabled={isSubmitting || taskState.isActive} /> </div>
                        <div> <Label>Evaluation Method <span className="text-red-500">*</span></Label> <RadioGroup value={evalChoice} onValueChange={handleEvalChoiceChange} className="flex space-x-4 mt-1" disabled={isSubmitting || taskState.isActive}> <div className="flex items-center space-x-2"> <RadioGroupItem value="standard" id="eval-standard" /> <Label htmlFor="eval-standard">Standard (MNIST)</Label> </div> <div className="flex items-center space-x-2"> <RadioGroupItem value="custom" id="eval-custom" /> <Label htmlFor="eval-custom">Upload Custom</Label> </div> </RadioGroup> </div>
                        {evalChoice === 'custom' && ( <div className="pl-2 pt-2 space-y-1"> <Label htmlFor="task-eval">Custom Evaluation Script (.py) <span className="text-red-500">*</span></Label> <Input ref={taskEvalRef} id="task-eval" type="file" accept=".py" required={evalChoice === 'custom'} onChange={handleTaskEvalFileChange} disabled={isSubmitting || taskState.isActive} /> </div> )}
                        <div> <Label htmlFor="init-weights">Initial Weights (.pth, Optional)</Label> <Input ref={weightsRef} id="init-weights" type="file" accept=".pth,.pt" onChange={handleWeightsFileChange} disabled={isSubmitting || taskState.isActive} /> </div>

                        {/* GA Parameter Inputs */}
                        <TooltipProvider delayDuration={100}>
                            <div className="space-y-5 border-t pt-6">
                                <h3 className="text-lg font-medium mb-4">GA Parameters</h3>
                                {gaParameterSchema.map((param: GaParameterSchemaItem) => {
                                    if (param.condition && !param.condition(formData)) return null;
                                    const inputId = `ga-${param.name}`;
                                    const displayValue = (param.type === 'boolean') ? !!formData[param.name] : (formData[param.name] ?? '');
                                    return (
                                        <div key={param.name} className="grid grid-cols-3 items-center gap-x-4 gap-y-1">
                                            {/* FIX: Wrap Label content in a span */}
                                            <Label htmlFor={inputId} className="col-span-1 flex items-start text-sm whitespace-nowrap">
                                                <span className="flex items-start"> {/* Use flex here to align text and icon */}
                                                    {param.label}
                                                    {param.description && (
                                                        <Tooltip>
                                                            <TooltipTrigger asChild>
                                                                <HelpCircle className="h-3.5 w-3.5 ml-1.5 text-muted-foreground hover:text-foreground cursor-help flex-shrink-0" />
                                                            </TooltipTrigger>
                                                            <TooltipContent side="right" className="max-w-xs text-xs" sideOffset={5}> <p>{param.description}</p> </TooltipContent>
                                                        </Tooltip>
                                                    )}
                                                </span>
                                            </Label>
                                            <div className="col-span-2">
                                                {/* Controls */}
                                                {param.type === 'boolean' && ( <div className="flex items-center pt-1"> <Switch id={inputId} name={param.name} checked={displayValue as boolean} onCheckedChange={(checked) => handleFormChange(param.name, checked)} disabled={isSubmitting || taskState.isActive} /> </div> )}
                                                {(param.type === 'number' || param.type === 'float') && ( <Input id={inputId} name={param.name} type="number" value={displayValue as string | number} onChange={(e: ChangeEvent<HTMLInputElement>) => handleFormChange(param.name, e.target.value)} min={param.min} max={param.max} step={param.step} required disabled={isSubmitting || taskState.isActive} className="w-full h-9 text-sm"/> )}
                                                {param.type === 'select' && param.options && ( <Select value={String(displayValue)} onValueChange={(value) => handleFormChange(param.name, value)} disabled={isSubmitting || taskState.isActive} name={param.name} required> <SelectTrigger id={inputId} className="w-full h-9 text-sm"><SelectValue placeholder="Select..." /></SelectTrigger> <SelectContent>{param.options.map(opt => (<SelectItem key={opt.value} value={String(opt.value)} className="text-sm">{opt.label}</SelectItem>))}</SelectContent> </Select> )}
                                                {param.type === 'slider' && ( <div className="flex items-center gap-2 pt-1"> <Slider id={inputId} name={param.name} value={[typeof formData[param.name] === 'number' ? formData[param.name] : param.defaultValue]} onValueChange={(value: number[]) => handleFormChange(param.name, value[0])} min={param.min} max={param.max} step={param.step} disabled={isSubmitting || taskState.isActive} className="flex-grow"/> <span className="text-xs text-muted-foreground w-12 text-right tabular-nums">{(typeof formData[param.name] === 'number' ? formData[param.name] : param.defaultValue).toFixed((param.step ?? 0.01) < 0.1 ? 2 : ((param.step ?? 1) < 1 ? 1 : 0))}</span> </div> )}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                         </TooltipProvider>

                        {/* Submit Button */}
                        <Button type="submit" className="w-full mt-6" disabled={isSubmitting || taskState.isActive || !modelDefFile || (evalChoice === 'custom' && !taskEvalFile)}>
                             {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                             {isSubmitting ? "Submitting..." : taskState.isActive ? "Task Running..." : "Start Evolution"}
                        </Button>
                    </form>
                </CardContent>
            </Card>

            {/* Status & Plot Card */}
            <Card>
                 <CardHeader> <CardTitle>Task Status & Results</CardTitle> <CardDescription>Monitor the evolution progress in real-time.</CardDescription> </CardHeader>
                <CardContent className="space-y-4">
                     {taskState.taskId ? (
                         <div key={taskState.taskId} className="space-y-2" >
                           <p>Task ID: <span className="font-mono text-sm bg-muted px-1 rounded">{taskState.taskId}</span></p>
                           <p>Status: <span className={`font-semibold ${taskState.status === 'SUCCESS' ? 'text-green-600' : taskState.status === 'FAILURE' ? 'text-red-600' : ''}`}>{taskState.status || 'N/A'}</span></p>
                           {(taskState.status === 'PROGRESS' || taskState.status === 'STARTED') && typeof taskState.progress === 'number' && ( <div className="pt-1"> <Progress value={taskState.progress * 100} className="w-full" /> <p className="text-sm text-muted-foreground pt-1">{Math.round(taskState.progress * 100)}% complete</p> </div> )}
                           {taskState.message && <p className="text-sm text-muted-foreground">{taskState.message}</p>}
                           {taskState.error && ( <Alert variant="destructive" className="mt-2"> <AlertCircle className="h-4 w-4" /> <AlertTitle>Task Error</AlertTitle> <AlertDescription>{taskState.error}</AlertDescription> </Alert> )}
                           {taskState.status === 'SUCCESS' && downloadLink !== undefined && (
                                <Button variant="outline" size="sm" asChild className="mt-2">
                                    {/* Wrap the anchor tag in a span to guarantee a single child element */}
                                    <span>
                                        <a href={downloadLink} download>Download Final Model (.pth)</a>
                                    </span>
                                </Button>
                           )}
                            {taskState.status === 'SUCCESS' && taskState.result?.fitness_history && (
                               <div className="mt-4 pt-4 border-t">
                                   <Button onClick={handleAnalyzeClick} disabled={ isAnalyzing || !taskState.result?.fitness_history || typeof formData.population_size !== 'number' || typeof formData.generations !== 'number' } variant="secondary" > {isAnalyzing ? ( <> <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing... </> ) : ( "Analyze Results with Gemini" )} </Button>
                                   {isAnalyzing && ( <p className="text-sm text-muted-foreground mt-2"> Contacting Gemini AI, this may take a moment... </p> )}
                                   {!analysisResult && !isAnalyzing && ( <p className="text-sm text-muted-foreground mt-2">Click "Analyze Results" to generate insights.</p> )}
                                   {analysisResult && !isAnalyzing && ( <Card className="mt-4 bg-muted/50"> <CardHeader className="pb-2 pt-4"> <CardTitle className="text-lg">Gemini Analysis</CardTitle> </CardHeader> <CardContent className="p-4 prose prose-sm max-w-none"> <ReactMarkdown>{analysisResult}</ReactMarkdown> </CardContent> </Card> )}
                               </div>
                           )}
                         </div>
                    ) : ( <p className="text-muted-foreground">Submit a task to see status and results.</p> )}
                     <div className="mt-4 h-72 border rounded bg-muted/20 flex items-center justify-center"> { hasPlotData ? ( <RealTimePlot maxFitnessData={plotData.maxFitness} avgFitnessData={plotData.avgFitness} diversityData={plotData.diversity}/> ) : ( <p className="text-muted-foreground">{taskState.taskId ? "Plot will appear here..." : "Submit task for plot"}</p> )} </div>
                </CardContent>
            </Card>
        </div>
    );
}
