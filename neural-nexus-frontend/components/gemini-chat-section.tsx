// components/gemini-chat-section.tsx
'use client';
import React, { useState, useRef, useEffect, FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import ReactMarkdown from 'react-markdown';
import { postGeminiQuery } from '@/lib/api';
import { Loader2, Send } from 'lucide-react';
import { toast } from 'sonner';

// Interface matching backend/Gemini format
interface GeminiHistoryItem {
    role: 'user' | 'model';
    parts: { text: string }[];
}

// Simple display message format
interface DisplayMessage {
    id: string; // For React key prop
    role: 'user' | 'model' | 'error';
    content: string;
}

export default function GeminiChatSection() {
    const [query, setQuery] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [geminiHistory, setGeminiHistory] = useState<GeminiHistoryItem[]>([]);
    const [displayHistory, setDisplayHistory] = useState<DisplayMessage[]>([]);
    const scrollAreaRef = useRef<HTMLDivElement>(null);

    // Function to scroll to bottom
    const scrollToBottom = () => {
        if (scrollAreaRef.current) {
            const scrollViewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
            if (scrollViewport) {
                scrollViewport.scrollTop = scrollViewport.scrollHeight;
            }
        }
    };

    // Scroll to bottom when displayHistory changes OR when loading starts
    useEffect(() => {
        setTimeout(scrollToBottom, 50);
    }, [displayHistory, isLoading]);

    const handleSendQuery = async (e?: FormEvent<HTMLFormElement>) => {
        e?.preventDefault();
        const trimmedQuery = query.trim();
        if (!trimmedQuery || isLoading) return;

        setIsLoading(true);
        const userQueryText = trimmedQuery;
        setQuery('');

        const userDisplayMessage: DisplayMessage = { id: `user-${Date.now()}`, role: 'user', content: userQueryText };
        setDisplayHistory(prev => [...prev, userDisplayMessage]);

        const historyToSend = [...geminiHistory];

        try {
            const response = await postGeminiQuery(userQueryText, historyToSend);
            const modelReply = response.reply;

            const modelDisplayMessage: DisplayMessage = { id: `model-${Date.now()}`, role: 'model', content: modelReply };
            setDisplayHistory(prev => [...prev, modelDisplayMessage]);

            setGeminiHistory(prev => [
                ...prev,
                { role: 'user', parts: [{ text: userQueryText }] },
                { role: 'model', parts: [{ text: modelReply }] }
            ]);

        } catch (error: any) {
            console.error("Gemini chat error:", error);
            const errorMessage: DisplayMessage = { id: `error-${Date.now()}`, role: 'error', content: `Error: ${error.message || 'Failed to get response'}` };
            setDisplayHistory(prev => [...prev, errorMessage]);
            toast.error(`Gemini Chat Error: ${error.message || 'Unknown error'}`);
        } finally {
            setIsLoading(false);
            setTimeout(() => document.getElementById('gemini-chat-input')?.focus(), 0);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendQuery();
        }
    };

    return (
        <Card className="h-full flex flex-col">
            <CardContent className="flex-grow p-4 overflow-hidden">
                 <ScrollArea className="h-[calc(100vh-220px)] pr-4" ref={scrollAreaRef}> {/* Adjust height if needed */}
                    <div className="space-y-4">
                        {displayHistory.map((message) => (
                            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`p-3 rounded-lg max-w-[75%] break-words ${
                                    message.role === 'user' ? 'bg-primary text-primary-foreground' : // User: Dark BG, Light Text
                                    message.role === 'model' ? 'bg-muted' : // Model: Muted BG (prose will handle text)
                                    'bg-destructive text-destructive-foreground' // Error: Destructive BG+Text
                                }`}>
                                    {/* FIX: Conditionally apply prose and ReactMarkdown ONLY for model messages */}
                                    {message.role === 'model' ? (
                                        <div className="prose prose-sm max-w-none">
                                            <ReactMarkdown>{message.content}</ReactMarkdown>
                                        </div>
                                    ) : (
                                        // For user and error messages, render text directly
                                        // This allows text-primary-foreground or text-destructive-foreground to apply
                                        <p>{message.content}</p>
                                    )}
                                </div>
                            </div>
                        ))}

                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="p-3 rounded-lg bg-muted flex items-center space-x-2 max-w-[75%]">
                                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                                    <span className="text-sm text-muted-foreground italic">
                                        Gemini is thinking...
                                    </span>
                                </div>
                            </div>
                        )}

                        {displayHistory.length === 0 && !isLoading && (
                            <p className="text-muted-foreground text-sm italic text-center">
                                No messages yet. Start the conversation!
                            </p>
                        )}

                    </div>
                 </ScrollArea>
            </CardContent>
            <CardFooter className="p-4 border-t">
                <form onSubmit={handleSendQuery} className="flex w-full items-center space-x-2">
                    <Input
                        id="gemini-chat-input"
                        placeholder="Ask Gemini anything..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        disabled={isLoading}
                        onKeyDown={handleKeyDown}
                        className="flex-1"
                        autoComplete="off"
                    />
                    <Button type="submit" disabled={isLoading || !query.trim()} size="icon">
                        {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                    </Button>
                </form>
            </CardFooter>
        </Card>
    );
}
