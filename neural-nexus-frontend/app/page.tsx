// app/page.tsx
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import AdvisorChat from "@/components/advisor-chat";
import EvolverSection from "@/components/evolver-section";
// Remove QuantizerSection import [2]
// import QuantizerSection from "@/components/quantizer-section";

export default function Home() {
  return (
    <div className="flex flex-col items-center w-full">
      <h1 className="text-3xl font-bold mb-6">Neural Nexus AI Platform</h1>
      <Tabs defaultValue="advisor" className="w-full max-w-4xl">
        {/* Adjust grid columns from 3 to 2 [2] */}
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="advisor">RAG Advisor</TabsTrigger>
          <TabsTrigger value="evolver">CNN/RNN Evolver</TabsTrigger>
          {/* Remove Quantizer Trigger [2] */}
          {/* <TabsTrigger value="quantizer">Quantizer</TabsTrigger> */}
        </TabsList>
        <TabsContent value="advisor" className="mt-4">
          <AdvisorChat />
        </TabsContent>
        <TabsContent value="evolver" className="mt-4">
          <EvolverSection />
        </TabsContent>
        {/* Remove Quantizer Content [2] */}
        {/*
        <TabsContent value="quantizer" className="mt-4">
           <QuantizerSection />
        </TabsContent>
        */}
      </Tabs>
    </div>
  );
}

