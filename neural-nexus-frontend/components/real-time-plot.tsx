// components/real-time-plot.tsx (Example using Recharts)
'use client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface PlotDataPoint {
    generation: number;
    fitness: number;
}

interface RealTimePlotProps {
    // Accept fitness history directly
    data: number[];
}

export default function RealTimePlot({ data }: RealTimePlotProps) {

    // Format data for Recharts
    const chartData: PlotDataPoint[] = data.map((fitness, index) => ({
        generation: index + 1, // Assuming generation starts at 1
        fitness: fitness,
    }));

    if (!chartData || chartData.length === 0) {
        return <p className="text-muted-foreground p-4">Waiting for data...</p>;
    }

    return (
        <ResponsiveContainer width="100%" height="100%">
            <LineChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="generation" label={{ value: 'Generation', position: 'insideBottomRight', offset: -5 }}/>
                <YAxis label={{ value: 'Best Fitness', angle: -90, position: 'insideLeft' }} domain={['auto', 'auto']}/>
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="fitness" stroke="#8884d8" activeDot={{ r: 8 }} />
            </LineChart>
        </ResponsiveContainer>
    );
}

// NOTE: For true real-time updates without full page re-renders on polling,
// you'd likely use WebSockets. The Celery task would push updates, a WebSocket
// client in React (perhaps in the useTaskPolling hook or a dedicated context)
// would receive them, and update the chart data state.
