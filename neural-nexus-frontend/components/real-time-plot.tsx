// components/real-time-plot.tsx
'use client'; // Required for client-side component
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Interface for the combined data point structure
interface PlotDataPoint {
    generation: number;
    maxFitness?: number | null; // Max fitness (can be null if data missing)
    avgFitness?: number | null; // Average fitness (can be null)
    diversity?: number | null;  // Diversity metric (can be null)
}

// Props interface accepting the different history arrays
interface RealTimePlotProps {
    maxFitnessData?: (number | null)[] | null; // Allow null arrays
    avgFitnessData?: (number | null)[] | null;
    diversityData?: (number | null)[] | null;
}

const RealTimePlot: React.FC<RealTimePlotProps> = ({
    maxFitnessData = [], // Default to empty array if null/undefined
    avgFitnessData = [],
    diversityData = []
}) => {
    // Ensure data are arrays
    const safeMaxFitness = Array.isArray(maxFitnessData) ? maxFitnessData : [];
    const safeAvgFitness = Array.isArray(avgFitnessData) ? avgFitnessData : [];
    const safeDiversity = Array.isArray(diversityData) ? diversityData : [];

    // Determine the maximum length based on available data
    const maxLen = Math.max(safeMaxFitness.length, safeAvgFitness.length, safeDiversity.length);

    if (maxLen === 0) {
        return <p className="text-sm text-muted-foreground p-4">Waiting for data...</p>;
    }

    // Combine data arrays into the structure Recharts expects
    const combinedData: PlotDataPoint[] = Array.from({ length: maxLen }, (_, i) => ({
        generation: i + 1, // Generation number (1-based index)
        maxFitness: safeMaxFitness[i] ?? null, // Use null if index out of bounds
        avgFitness: safeAvgFitness[i] ?? null,
        diversity: safeDiversity[i] ?? null,
    }));

    // Check if we actually have data for each line to avoid rendering empty lines/axes
    const hasMaxFitness = safeMaxFitness.some(d => d !== null && !isNaN(d));
    const hasAvgFitness = safeAvgFitness.some(d => d !== null && !isNaN(d));
    const hasDiversity = safeDiversity.some(d => d !== null && !isNaN(d));

    return (
        <ResponsiveContainer width="100%" height="100%">
            <LineChart
                data={combinedData}
                margin={{ top: 5, right: 35, left: 5, bottom: 15 }} // Adjust margins for labels
            >
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                {/* X-Axis: Generation */}
                <XAxis
                    dataKey="generation"
                    type="number" // Ensure it treats generation as a number
                    domain={['dataMin', 'dataMax']} // Use actual data range
                    allowDecimals={false}
                    label={{ value: 'Generation', position: 'insideBottom', offset: -5, fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                    stroke="hsl(var(--muted-foreground))"
                    tick={{ fontSize: 10 }}
                />
                {/* Y-Axis 1 (Left): Fitness */}
                <YAxis
                    yAxisId="fitnessAxis" // ID for this axis
                    label={{ value: 'Fitness', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                    stroke="hsl(var(--muted-foreground))"
                    tick={{ fontSize: 10 }}
                    domain={['auto', 'auto']} // Auto-scale fitness axis
                    allowDataOverflow={true} // Allows lines to slightly exceed axis domain if needed
                />
                {/* Y-Axis 2 (Right): Diversity (Conditional) */}
                {hasDiversity && (
                    <YAxis
                        yAxisId="diversityAxis" // ID for the second axis
                        orientation="right"
                        label={{ value: 'Diversity', angle: 90, position: 'insideRight', offset: 10, fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                        stroke="hsl(var(--muted-foreground))"
                        tick={{ fontSize: 10 }}
                        domain={['auto', 'auto']} // Auto-scale diversity axis
                        allowDataOverflow={true}
                    />
                )}

                <Tooltip
                    contentStyle={{
                        backgroundColor: 'hsl(var(--background))',
                        borderColor: 'hsl(var(--border))',
                        borderRadius: 'var(--radius)',
                        fontSize: '12px',
                        padding: '5px 10px',
                    }}
                    labelStyle={{ color: 'hsl(var(--foreground))', marginBottom: '5px' }}
                    itemStyle={{ color: 'hsl(var(--foreground))', fontSize: '11px' }}
                    formatter={(value: number) => typeof value === 'number' ? value.toFixed(4) : value} // Format tooltip values
                />
                <Legend verticalAlign="top" height={30} wrapperStyle={{fontSize: '11px'}}/>

                {/* --- Line Components --- */}
                {hasMaxFitness && (
                    <Line
                        yAxisId="fitnessAxis" // Assign to the left axis
                        type="monotone"
                        dataKey="maxFitness"
                        name="Max Fitness"
                        stroke="hsl(var(--primary))" // Example color
                        strokeWidth={2}
                        dot={false}
                        connectNulls // Connect line even if some data points are missing
                    />
                )}
                {hasAvgFitness && (
                    <Line
                        yAxisId="fitnessAxis" // Assign to the left axis
                        type="monotone"
                        dataKey="avgFitness"
                        name="Avg Fitness"
                        stroke="hsl(var(--primary) / 0.6)" // Example: Slightly faded primary
                        strokeWidth={1.5}
                        strokeDasharray="4 4" // Dashed line
                        dot={false}
                        connectNulls
                    />
                )}
                {hasDiversity && (
                    <Line
                        yAxisId="diversityAxis" // Assign to the right axis
                        type="monotone"
                        dataKey="diversity"
                        name="Diversity"
                        stroke="hsl(var(--secondary-foreground))" // Example: Secondary color
                        strokeWidth={1.5}
                        dot={false}
                        connectNulls
                    />
                )}
            </LineChart>
        </ResponsiveContainer>
    );
};

export default RealTimePlot;
