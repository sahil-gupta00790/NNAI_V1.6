// app/page.tsx
'use client'; // Add if not already present, needed for Tabs state

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
// Rename AdvisorChat import to match convention if needed
// import AdvisorChat from "@/components/advisor-chat";
import RagAdvisorSection from "@/components/advisor-chat"; // Assuming this is the correct name now
import EvolverSection from "@/components/evolver-section";
import GeminiChatSection from "@/components/gemini-chat-section";

export default function Home() {
  return (
    // Using container for max-width and centering, px for padding
    <main className="container mx-auto px-4 py-8 flex flex-col items-center">
      <div className="text-center mb-8"> {/* Centering container for title and subtitle */}
        <h1 className="text-3xl sm:text-4xl font-bold mb-2"> {/* Slightly adjust bottom margin */}
          NeuroForge
        </h1>
        {/* --- ADDED SUBTITLE --- */}
        <p className="text-sm sm:text-base text-muted-foreground">
          Evolving Intelligence, Intelligently.
          {/* Other options: */}
          {/* Where Evolution Meets Intelligence. */}
          {/* Shape the Future of Neural Networks. */}
          {/* Your AI Evolution Workbench. */}
        </p>
        {/* --- END SUBTITLE --- */}
      </div>

      {/* Tabs Component */}
      <Tabs defaultValue="evolver" className="w-full max-w-5xl"> {/* Increased max-width slightly */}
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="advisor">RAG Advisor</TabsTrigger>
          <TabsTrigger value="evolver">CNN/RNN Evolver</TabsTrigger>
          <TabsTrigger value="gemini_chat">Direct Chat</TabsTrigger>
        </TabsList>

        {/* Tab Content */}
        <TabsContent value="advisor" className="mt-4">
          {/* Ensure correct component name */}
          <RagAdvisorSection />
          {/* <AdvisorChat /> */}
        </TabsContent>
        <TabsContent value="evolver" className="mt-4">
          <EvolverSection />
        </TabsContent>
        <TabsContent value="gemini_chat" className="mt-4">
           <GeminiChatSection />
        </TabsContent>
      </Tabs>
    </main>
  );
}
