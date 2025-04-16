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
import { startEvolutionTask, analyzeGaResults } from '@/lib/api';
import { useTaskPolling } from '@/lib/hooks/useTaskPolling';
import RealTimePlot from './real-time-plot';
import { toast } from "sonner";
import { motion } from "framer-motion";
import ReactMarkdown from 'react-markdown';

// --- Define the schema for GA parameters ---
const gaParameterSchema = [
  { name: 'generations', label: 'Generations', type: 'number', defaultValue: 20, min: 1, max: 500, step: 1, description: 'Number of generations the evolution will run.' },
  { name: 'population_size', label: 'Population Size', type: 'number', defaultValue: 30, min: 2, max: 500, step: 1, description: 'Number of individuals in each generation.' },
  { name: 'selection_strategy', label: 'Selection Strategy', type: 'select', defaultValue: 'tournament', options: [ { value: 'tournament', label: 'Tournament' }, { value: 'roulette', label: 'Roulette Wheel' } ], description: 'Method for selecting parents for reproduction.'},
  { name: 'tournament_size', label: 'Tournament Size', type: 'number', defaultValue: 3, min: 2, max: 20, step: 1, condition: (formData: Record<string, any>) => formData.selection_strategy === 'tournament', description: 'Number of individuals competing in each tournament selection.' }, // Conditional field
  { name: 'crossover_operator', label: 'CrossoverOperator', type: 'select', defaultValue: 'one_point', options: [ { value: 'one_point', label: 'One Point' }, { value: 'uniform', label: 'Uniform' }, { value: 'average', label: 'Average' } ], description: 'Method for combining parent weights to create offspring.' },
  { name: 'uniform_crossover_prob', label: 'Uniform Crossover Prob', type: 'slider', defaultValue: 0.5, min: 0, max: 1, step: 0.01, condition: (formData: Record<string, any>) => formData.crossover_operator === 'uniform', description: 'Probability of swapping genes between parents in uniform crossover.' }, // Conditional field
  { name: 'mutation_operator', label: 'Mutation Operator', type: 'select', defaultValue: 'gaussian', options: [ { value: 'gaussian', label: 'Gaussian Noise' }, { value: 'uniform_random', label: 'Uniform Random Replacement' } ], description: 'Method for introducing random changes to offspring weights.' },
  { name: 'mutation_rate', label: 'Mutation Rate', type: 'slider', defaultValue: 0.15, min: 0, max: 1, step: 0.01, description: 'Probability of each weight being mutated.' },
  { name: 'mutation_strength', label: 'Mutation Strength', type: 'float', defaultValue: 0.05, min: 0, step: 0.001, description: 'Magnitude of mutation (e.g., std dev for Gaussian). Use eval_config for uniform range details.' }, // Input for strength/std dev
  { name: 'elitism_count', label: 'Elitism Count', type: 'number', defaultValue: 1, min: 0, max: 10, step: 1, description: 'Number of best individuals carried directly to the next generation.' },
  { name: 'eval_batch_size', label: 'Eval Batch Size', type: 'number', defaultValue: 128, min: 1, max: 1024, step: 1, description: 'Batch size used during model evaluation (passed in eval_config).'}
];
// --- End Schema ---

// --- Helper to initialize state from schema ---
function getInitialFormData() {
  const initialData: Record<string, any> = {};
  gaParameterSchema.forEach(param => { initialData[param.name] = param.defaultValue; });
  initialData['model_class'] = 'MyCNN';
  initialData['model_args'] = [];
  initialData['model_kwargs'] = {};
  initialData['eval_config'] = { batch_size: initialData['eval_batch_size'] };
  delete initialData['eval_batch_size'];
  return initialData;
}
// --- End Helper ---

// --- Interface for context passed to analysis ---
interface ConfigContextForAnalysis {
  generations?: number;
  population_size?: number;
  mutation_rate?: number;
  mutation_strength?: number;
  // Add other config fields if you want to pass them from formData
}
// --- End Interface ---

export default function EvolverSection() {
    // --- State ---
    const [modelDefFile, setModelDefFile] = useState<File | null>(null);
    const [taskEvalFile, setTaskEvalFile] = useState<File | null>(null);
    const [weightsFile, setWeightsFile] = useState<File | null>(null);
    // Use formData state instead of configJson for UI editing
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
        // Ensure numbers/floats from inputs/sliders are stored as numbers
        const schemaItem = gaParameterSchema.find(p => p.name === name);
        if (schemaItem && (schemaItem.type === 'number' || schemaItem.type === 'float' || schemaItem.type === 'slider')) {
            // Allow empty string to temporarily clear input, convert to null for state
            processedValue = value === '' ? null : parseFloat(value);
            // If parseFloat results in NaN (e.g., from non-numeric input), keep it null
            if (isNaN(processedValue as number)) {
                 processedValue = null;
            }
        }

        // Special handling for eval_batch_size to update nested eval_config
        if (name === 'eval_batch_size') {
             setFormData(prev => ({
                 ...prev,
                 [name]: value === '' ? '' : processedValue,
                 eval_config: { // Update nested structure
                    ...(prev.eval_config || {}), // Preserve other keys
                    batch_size: typeof processedValue === 'number' ? processedValue : undefined
                 }
             }));
        } else {
             setFormData(prev => ({
                 ...prev,
                 [name]: processedValue // Use processed value (number, string, or null)
             }));
        }
    };
    // --- End Change Handler ---

    // Event Handlers (Unchanged)
    const handleEvalChoiceChange = (value: 'standard' | 'custom') => {
         setEvalChoice(value);
         if (value === 'standard' && taskEvalRef.current) {
             taskEvalRef.current.value = ""; setTaskEvalFile(null);
         }
    };

    const handleModelDefFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] ?? null;
        setModelDefFile(file);
    };

    const handleTaskEvalFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] ?? null;
        setTaskEvalFile(file);
    };

    const handleWeightsFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] ?? null;
        setWeightsFile(file);
    };

    // handleSubmit (MODIFIED)
    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        // Validations
        if (!modelDefFile) { toast.error("Model Definition file is required."); return; }
        if (evalChoice === 'custom' && !taskEvalFile) { toast.error("Custom Evaluation Script file is required."); return; }
        if (taskState.isActive) { toast.warning("A task is already running."); return; }

        // --- Prepare and Validate FormData ---
        const finalFormDataObject = { ...formData };

        // Remove temporary UI keys if they exist before stringifying
        delete finalFormDataObject['eval_batch_size'];

        // Validate required fields before sending (more specific checks)
        const requiredNumericFields = ['generations', 'population_size', 'elitism_count'];
        for (const field of requiredNumericFields) {
            if (typeof finalFormDataObject[field] !== 'number' || finalFormDataObject[field] < (schemaItem => schemaItem?.min ?? 0)(gaParameterSchema.find(p => p.name === field))) {
                 const schemaItem = gaParameterSchema.find(p => p.name === field);
                 toast.error(`'${schemaItem?.label || field}' must be a valid number${schemaItem?.min !== undefined ? ` >= ${schemaItem.min}` : ''}.`);
                 return;
            }
        }
        // Validate mutation params
         if (typeof finalFormDataObject['mutation_rate'] !== 'number' || finalFormDataObject['mutation_rate'] < 0 || finalFormDataObject['mutation_rate'] > 1) { toast.error("'Mutation Rate' must be a number between 0 and 1."); return; }
         if (typeof finalFormDataObject['mutation_strength'] !== 'number' || finalFormDataObject['mutation_strength'] < 0) { toast.error("'Mutation Strength' must be a non-negative number."); return; }
        // Ensure model_class is present (should be from initial state)
        if (!finalFormDataObject.model_class || typeof finalFormDataObject.model_class !== 'string' || finalFormDataObject.model_class.trim() === "") {
            toast.error("Internal Error: Model class configuration is missing."); return;
        }

        // Convert the validated formData object to JSON string for the API
        const configJsonString = JSON.stringify(finalFormDataObject, null, 2); // Pretty print optional
        console.log("Frontend: Sending final config JSON string:", configJsonString);
        // --- END MODIFICATION ---

        setIsSubmitting(true);
        resetTaskState();
        setAnalysisResult(null);
        setIsAnalyzing(false);
        toast("Submitting evolution task...");

        // FormData for API (Sends JSON string)
        const apiFormData = new FormData();
        apiFormData.append('model_definition', modelDefFile);
        apiFormData.append('use_standard_eval', String(evalChoice === 'standard'));
        if (evalChoice === 'custom' && taskEvalFile) apiFormData.append('task_evaluation', taskEvalFile);
        if (weightsFile) apiFormData.append('initial_weights', weightsFile);
        apiFormData.append('config_json', configJsonString); // Send the stringified config

        try {
            const response = await startEvolutionTask(apiFormData);
            startTask(response.task_id);
            toast.success(`Task ${response.task_id} started.`);
             // Reset file inputs, etc.
             if(modelDefRef.current) modelDefRef.current.value = "";
             if(taskEvalRef.current) taskEvalRef.current.value = "";
             if(weightsRef.current) weightsRef.current.value = "";
             setModelDefFile(null); setTaskEvalFile(null); setWeightsFile(null);
             // Optionally reset formData to defaults: setFormData(getInitialFormData());
        } catch (error: any) {
            console.error("Error starting evolution task:", error);
            toast.error(`Failed to start task: ${error.message || 'Unknown error'}`);
            resetTaskState(); // Also reset task state on submission failure
        } finally {
            setIsSubmitting(false);
        }
    };
    // --- End handleSubmit ---

    // handleAnalyzeClick (No changes - trailing comma is fine)
    const handleAnalyzeClick = async () => {
        if (taskState.status !== 'SUCCESS' || !taskState.result || !Array.isArray(taskState.result.fitness_history)) { toast.error("Task not successfully completed or result data is missing/invalid for analysis."); return; }
        setIsAnalyzing(true);
        setAnalysisResult(null);
        try {
            // Use current formData state for config context
            const configContext: ConfigContextForAnalysis = formData; // Cast or use directly

<<<<<<< HEAD
            const analysisPayload = {
                fitness_history: taskState.result.fitness_history,
                avg_fitness_history: taskState.result.avg_fitness_history ?? null,
                diversity_history: taskState.result.diversity_history ?? null,
                // Safely access config context with defaults
                generations: typeof configContext?.generations === 'number' ? configContext.generations : taskState.result.fitness_history.length,
                population_size: typeof configContext?.population_size === 'number' ? configContext.population_size : 0,
                mutation_rate: typeof configContext?.mutation_rate === 'number' ? configContext.mutation_rate : undefined,
                mutation_strength: typeof configContext?.mutation_strength === 'number' ? configContext.mutation_strength : undefined,
=======
            let generations: number | undefined = typeof formData.generations === 'number' ? formData.generations : undefined;
            if (!generations || generations <= 0) {
                const historyLength = fitnessHistory?.length;
                if (historyLength && historyLength > 0) { generations = historyLength; }
                else { toast.error("Invalid or missing generation count. Cannot perform analysis."); setIsAnalyzing(false); return; }
            }

            const populationSize: number | undefined = typeof formData.population_size === 'number' ? formData.population_size : undefined;
             if (typeof populationSize !== 'number') {
                 toast.error("Population size is missing or invalid in the configuration. Cannot perform analysis.");
                 setIsAnalyzing(false); return;
             }

             const mutationRate: number | undefined = typeof formData.mutation_rate === 'number' ? formData.mutation_rate : undefined;
             const mutationStrength: number | undefined = typeof formData.mutation_strength === 'number' ? formData.mutation_strength : undefined;

            const finalPayload = {
                fitness_history: fitnessHistory,
                avg_fitness_history: avgFitnessHistory,
                diversity_history: diversityHistory,
                generations: generations,
                population_size: populationSize,
                mutation_rate: mutationRate,
                mutation_strength: mutationStrength, // Trailing comma is fine
            };
            if (analysisPayload.generations <= 0) { throw new Error("Invalid generation count for analysis."); }
            toast.info("Requesting analysis from Gemini AI...");
            const response = await analyzeGaResults(analysisPayload);
            setAnalysisResult(response.analysis_text);
            toast.success("Analysis received!");
        } catch (error: any) { console.error("Error fetching analysis:", error); toast.error(`Failed to generate analysis: ${error.message || 'Unknown error'}`); } finally { setIsAnalyzing(false); }
    };
    // --- End handleAnalyzeClick ---

    // Plot Data Prep, Download Link (Unchanged)
    const plotData = {
        maxFitness: Array.isArray(taskState.fitnessHistory) ? taskState.fitnessHistory : [],
        avgFitness: Array.isArray(taskState.avgFitnessHistory) ? taskState.avgFitnessHistory : [],
        diversity: Array.isArray(taskState.diversityHistory) ? taskState.diversityHistory : []
    };
    const hasPlotData = plotData.maxFitness.length > 0 || plotData.avgFitness.length > 0 || plotData.diversity.length > 0;
    const downloadLink = taskState.status === 'SUCCESS' && taskState.result?.final_model_path && taskState.taskId
        ? `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/evolver/results/${taskState.taskId}/download`
        : undefined;

    // --- Component Return ---
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Form Card */}
            <Card>
                <CardHeader>
                    <CardTitle>Configure Evolution</CardTitle>
                    <CardDescription>Set files and Genetic Algorithm parameters.</CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-6"> {/* Increased vertical spacing */}
                        {/* --- File Inputs (Structure Unchanged) --- */}
                         <div>
                            <Label htmlFor="model-def">Model Definition (.py) <span className="text-red-500">*</span></Label>
                            <Input ref={modelDefRef} id="model-def" type="file" accept=".py" required onChange={handleModelDefFileChange} disabled={isSubmitting || taskState.isActive} />
                        </div>
                        <div>
                            <Label>Evaluation Method <span className="text-red-500">*</span></Label>
                            <RadioGroup value={evalChoice} onValueChange={handleEvalChoiceChange} className="flex space-x-4 mt-1" disabled={isSubmitting || taskState.isActive}>
                                 <div className="flex items-center space-x-2"> <RadioGroupItem value="standard" id="eval-standard" /> <Label htmlFor="eval-standard">Standard (MNIST)</Label> </div>
                                 <div className="flex items-center space-x-2"> <RadioGroupItem value="custom" id="eval-custom" /> <Label htmlFor="eval-custom">Upload Custom</Label> </div>
                            </RadioGroup>
                        </div>
                        {/* {<AnimatePresence> */}
                            {evalChoice === 'custom' && (
                                //  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.3 }} className="mt-2">
                                    <>
                                        <Label htmlFor="task-eval">Custom Evaluation Script (.py) <span className="text-red-500">*</span></Label>
                                        <Input ref={taskEvalRef} id="task-eval" type="file" accept=".py" required={evalChoice === 'custom'} onChange={handleTaskEvalFileChange} disabled={isSubmitting || taskState.isActive} />
                                    </>
                                //  </motion.div>
                             )}
                         {/* </AnimatePresence> */}
                        <div>
                            <Label htmlFor="init-weights">Initial Weights (.pth, Optional)</Label>
                            <Input ref={weightsRef} id="init-weights" type="file" accept=".pth,.pt" onChange={handleWeightsFileChange} disabled={isSubmitting || taskState.isActive} />
                        </div>
                        {/* --- END File Inputs --- */}

                        {/* --- Dynamically Generated GA Parameter Inputs --- */}
                        <TooltipProvider delayDuration={100}>
                            <div className="space-y-5 border-t pt-6"> {/* Use more vertical space */}
                                <h3 className="text-lg font-medium mb-4">GA Parameters</h3>
                                {gaParameterSchema.map((param) => {
                                    // Check conditional rendering
                                    if (param.condition && !param.condition(formData)) {
                                        return null; // Don't render if condition not met
                                    }

                                    const inputId = `ga-${param.name}`;
                                    const displayValue = formData[param.name] ?? '';

                                    return (
                                        <div key={param.name} className="grid grid-cols-3 items-center gap-x-4 gap-y-1">
                                            <Label htmlFor={inputId} className="col-span-1 flex items-start text-sm whitespace-nowrap">
                                               {/* FIX: Wrap adjacent label text and tooltip in a Span */}
                                                
                                                    {param.label}
                                                    {param.description && (
                                                        <Tooltip>
                                                            <TooltipTrigger asChild>
                                                                {/* Single child for TooltipTrigger */}
                                                                <HelpCircle className="h-3.5 w-3.5 ml-1.5 text-muted-foreground hover:text-foreground cursor-help" />
                                                            </TooltipTrigger>
                                                            <TooltipContent side="right" className="max-w-xs text-xs" sideOffset={5}>
                                                                <p>{param.description}</p>
                                                            </TooltipContent>
                                                        </Tooltip>
                                                    )}
                                                
                                            </Label>
                                            <div className="col-span-2">
                                                {/* Input controls - No changes needed here */}
                                                {(param.type === 'number' || param.type === 'float') && (
                                                    <Input
                                                        id={inputId}
                                                        name={param.name}
                                                        type="number"
                                                        value={displayValue} // Use potentially empty string for display
                                                        onChange={(e: ChangeEvent<HTMLInputElement>) => handleFormChange(param.name, e.target.value)} // Pass raw value
                                                        min={param.min}
                                                        max={param.max}
                                                        step={param.step}
                                                        required // Mark as required
                                                        disabled={isSubmitting || taskState.isActive}
                                                        className="w-full h-9 text-sm"
                                                    />
                                                )}
                                                {param.type === 'select' && param.options && (
                                                     <Select
                                                         value={String(displayValue)} // Ensure string value for Select state
                                                         onValueChange={(value) => handleFormChange(param.name, value)} // String value from select
                                                         disabled={isSubmitting || taskState.isActive}
                                                         name={param.name}
                                                         required
                                                     >
                                                         <SelectTrigger id={inputId} className="w-full h-9 text-sm">
                                                             <SelectValue placeholder="Select..." />
                                                         </SelectTrigger>
                                                         <SelectContent>
                                                             {param.options.map(opt => (
                                                                 <SelectItem key={opt.value} value={String(opt.value)} className="text-sm">{opt.label}</SelectItem>
                                                             ))}
                                                         </SelectContent>
                                                     </Select>
                                                )}
                                                 {param.type === 'slider' && (
                                                    <div className="flex items-center gap-2 pt-1">
                                                        <Slider
                                                             id={inputId}
                                                             name={param.name}
                                                             // Ensure slider value is always a number array, use default if formData value is invalid/null
                                                             value={[typeof formData[param.name] === 'number' ? formData[param.name] : param.defaultValue]}
                                                             onValueChange={(value: number[]) => handleFormChange(param.name, value[0])}
                                                             min={param.min}
                                                             max={param.max}
                                                             step={param.step}
                                                             disabled={isSubmitting || taskState.isActive}
                                                             className="flex-grow"
                                                         />
                                                        <span className="text-xs text-muted-foreground w-12 text-right tabular-nums">
                                                             {(typeof formData[param.name] === 'number'
                                                               ? formData[param.name]
                                                               : param.defaultValue
                                                             ).toFixed((param.step ?? 0.01) < 0.1 ? 2 : ((param.step ?? 1) < 1 ? 1 : 0))}
                                                         </span>
                                                    </div>
                                                 )}
                                                 {/* Add Switch or Checkbox here if needed for boolean params */}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                         </TooltipProvider>
                        {/* --- End Dynamic Inputs --- */}

                        {/* Submit Button */}
                        <Button type="submit" className="w-full mt-6" disabled={isSubmitting || taskState.isActive || !modelDefFile || (evalChoice === 'custom' && !taskEvalFile)}>
                            {isSubmitting ? "Submitting..." : taskState.isActive ? "Task Running..." : "Start Evolution"}
                        </Button>
                    </form>
                </CardContent>
            </Card>

            {/* Status & Plot Card (Structure Unchanged) */}
            <Card>
                 <CardHeader>
                     <CardTitle>Task Status & Results</CardTitle>
                     <CardDescription>Monitor the evolution progress in real-time.</CardDescription>
                 </CardHeader>
                <CardContent className="space-y-4">
                     {/* Status Display Block */}
                    {taskState.taskId ? (
                         <div key={taskState.taskId} className="space-y-2" >
                           {/* Status Info */}
                           <p>Task ID: <span className="font-mono text-sm bg-muted px-1 rounded">{taskState.taskId}</span></p>
                           <p>Status: <span className={`font-semibold ${taskState.status === 'SUCCESS' ? 'text-green-600' : taskState.status === 'FAILURE' ? 'text-red-600' : ''}`}>{taskState.status || 'N/A'}</span></p>
                           {(taskState.status === 'PROGRESS' || taskState.status === 'STARTED') && typeof taskState.progress === 'number' && ( <div className="pt-1"> <Progress value={taskState.progress * 100} className="w-full" /> <p className="text-sm text-muted-foreground pt-1">{Math.round(taskState.progress * 100)}% complete</p> </div> )}
                           {taskState.message && <p className="text-sm text-muted-foreground">{taskState.message}</p>}
                           {taskState.error && ( <Alert variant="destructive" className="mt-2"> <AlertCircle className="h-4 w-4" /> <AlertTitle>Task Error</AlertTitle> <AlertDescription>{taskState.error}</AlertDescription> </Alert> )}

                           {/* FIX: Download Button - Ensure <a> is the DIRECT child */}
                           {taskState.status === 'SUCCESS' && downloadLink !== undefined && (
                                <Button variant="outline" size="sm" asChild className="mt-2">
                                    {/* Removed the extra <span> wrap */}
                                    <a href={downloadLink} download>Download Final Model (.pth)</a>
                                </Button>
                           )}

                           {/* Gemini Analysis Section */}
                           {taskState.status === 'SUCCESS' && taskState.result?.fitness_history && (
                               <div className="mt-4 pt-4 border-t">
                                   <Button onClick={handleAnalyzeClick} disabled={ isAnalyzing || !taskState.result?.fitness_history || typeof formData.population_size !== 'number' || typeof formData.generations !== 'number' } variant="secondary" > {isAnalyzing ? ( <> <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing... </> ) : ( "Analyze Results with Gemini" )} </Button>
                                   {isAnalyzing && ( <p className="text-sm text-muted-foreground mt-2"> Contacting Gemini AI, this may take a moment... </p> )}
                                   {/* ADD A PLACEHOLDER COMPONENT HERE */}
                                   {!analysisResult && !isAnalyzing && (
                                        <p className="text-sm text-muted-foreground mt-2">No analysis available.</p>
                                   )}
                                   {analysisResult && !isAnalyzing && (
                                       <Card className="mt-4 bg-muted/50">
                                            <CardHeader className="pb-2 pt-4"> <CardTitle className="text-lg">Gemini Analysis</CardTitle> </CardHeader>
                                            {/* Ensure prose fix is applied */}
                                           <CardContent className="p-4 prose prose-sm max-w-none">
                                                <ReactMarkdown>{analysisResult}</ReactMarkdown>
                                           </CardContent>
                                        </Card>
                                   )}
                               </div>
                           )}
                           {/* End Gemini Analysis Section */}
                         </div>
                    ) : ( <p className="text-muted-foreground">Submit a task to see status and results.</p> )}
                     {/* End Status Block */}

                     {/* Plot Area */}
                     <div className="mt-4 h-72 border rounded bg-muted/20 flex items-center justify-center"> { hasPlotData ? ( <RealTimePlot maxFitnessData={plotData.maxFitness} avgFitnessData={plotData.avgFitness} diversityData={plotData.diversity}/> ) : ( <p className="text-muted-foreground">{taskState.taskId ? "Plot will appear here..." : "Submit task for plot"}</p> )} </div>
                </CardContent>
            </Card>
        </div>
    );
}
