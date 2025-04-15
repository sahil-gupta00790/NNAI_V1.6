// app/layout.tsx (Example with Theme Toggle using Shadcn)
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider"; // Assuming you add theme toggle
import { Toaster } from "sonner"; // For notifications

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Neural Nexus AI",
  description: "Integrated AI Development Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <main className="container mx-auto p-4">{children}</main>
          <Toaster richColors /> {/* For toast notifications */}
        </ThemeProvider>
      </body>
    </html>
  );
}
