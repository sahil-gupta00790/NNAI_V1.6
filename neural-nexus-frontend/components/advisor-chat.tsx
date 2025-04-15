// components/advisor-chat.tsx
'use client';
import { useState, useEffect, useRef } from 'react'; // Added useEffect, useRef
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { postAdvisorQuery } from '@/lib/api'; // Assuming this returns { response: string, sources?: string[] }
import { toast } from "sonner";
import { motion } from "framer-motion";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ScrollArea } from "@/components/ui/scroll-area"; // Keep ScrollArea import

interface Message {
    id: number; // Add unique ID for stable key
    sender: 'user' | 'ai';
    text: string;
    sources?: string[]; // Optional sources array for AI messages
}

export default function AdvisorChat() {
    const [query, setQuery] = useState('');
    const [history, setHistory] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [messageIdCounter, setMessageIdCounter] = useState(0); // Counter for unique message IDs
    const scrollAreaRef = useRef<HTMLDivElement>(null); // Ref for scrolling control

    // Function to scroll to bottom
    const scrollToBottom = () => {
      const scrollViewport = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollViewport) {
        setTimeout(() => { // Timeout allows render to complete before scrolling
            scrollViewport.scrollTop = scrollViewport.scrollHeight;
        }, 0);
      }
    };

    // Scroll to bottom whenever history changes
    useEffect(() => {
      scrollToBottom();
    }, [history]); // Dependency array includes history

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim() || isLoading) return;

        const nextId = messageIdCounter;
        const userMessage: Message = { id: nextId, sender: 'user', text: query };
        setHistory(prev => [...prev, userMessage]);
        setQuery('');
        setIsLoading(true);
        setMessageIdCounter(prev => prev + 2); // Increment counter

        try {
            const response = await postAdvisorQuery(query); // Expects { response: string, sources?: string[] }

            const aiMessage: Message = {
                id: nextId + 1, // Assign ID
                sender: 'ai',
                text: response.response || "Sorry, I couldn't generate a response.",
                sources: response.sources // Store sources with the message
            };
            setHistory(prev => [...prev, aiMessage]);
        } catch (error: any) {
            console.error("Error fetching advisor response:", error);
             const errorMessage: Message = {
                id: nextId + 1, // Assign ID
                sender: 'ai',
                text: `Sorry, I encountered an error retrieving the response: ${error.message || 'Unknown error'}`
             };
            setHistory(prev => [...prev, errorMessage]);
            toast.error("Failed to get response from advisor.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Card>
            <CardHeader>
                <CardTitle>Neural Network Advisor</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                {/* Chat History Area using ScrollArea */}
                <ScrollArea className="h-96 w-full rounded-md border" ref={scrollAreaRef}> {/* Added ref */}
                   <div className="p-4 space-y-3 flex flex-col bg-muted/30 min-h-full"> {/* Added min-h-full */}
                        {history.map((msg) => ( // Use msg.id as key
                            <motion.div
                                key={msg.id} // Use stable ID
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.3 }}
                                className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div
                                    className={`p-3 rounded-lg max-w-[85%] break-words shadow-sm ${
                                        msg.sender === 'user' ? 'bg-primary text-primary-foreground' : 'bg-card border'
                                    }`}
                                >
                                    {msg.sender === 'ai' ? (
                                        <>
                                            {/* AI Message Content */}
                                            <div className="prose prose-sm dark:prose-invert max-w-none text-foreground"> {/* Added text-foreground */}
                                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                    {msg.text}
                                                </ReactMarkdown>
                                            </div>
                                            {/* Source Document Display */}
                                            {msg.sources && msg.sources.length > 0 && (
                                                <div className="mt-2 pt-2 border-t border-border/50"> {/* Use theme border */}
                                                    <p className="text-xs font-semibold text-muted-foreground mb-1">Sources:</p>
                                                    <ul className="list-disc list-inside text-xs text-muted-foreground space-y-0.5">
                                                        {msg.sources.map((source, i) => (
                                                            <li key={i}>{source}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </>
                                    ) : (
                                        // User Message Content
                                        msg.text
                                    )}
                                </div>
                            </motion.div>
                        ))}
                        {/* Loading Indicator */}
                        {isLoading && history[history.length - 1]?.sender === 'user' && (
                            <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
                                <div className="p-3 rounded-lg max-w-[85%] bg-muted animate-pulse"> Thinking... </div>
                            </motion.div>
                        )}
                   </div> {/* Inner div for padding and flex */}
                </ScrollArea> {/* End ScrollArea */}

                {/* Input Form */}
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <Input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask about NN architecture, evolution, quantization..."
                        disabled={isLoading}
                        autoComplete="off"
                    />
                    <Button type="submit" disabled={isLoading || !query.trim()}>
                        {isLoading ? "Sending..." : "Send"}
                    </Button>
                </form>
            </CardContent>
        </Card>
    );
}
