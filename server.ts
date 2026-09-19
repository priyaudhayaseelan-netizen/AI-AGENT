import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;

let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is missing.");
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

export interface QuizQuestion {
  question: string;
  options: string[];
  correctAnswerIndex: number;
  explanation: string;
  difficulty: string;
  topic: string;
}

async function startServer() {
  const app = express();
  app.use(express.json());

  // Health check endpoint
  app.get("/api/health", (_req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });

  // Generate quiz question endpoint
  app.post("/api/quiz/generate", async (req, res) => {
    try {
      const { topic, difficulty = "medium", previousQuestions = [] } = req.body;

      if (!topic || typeof topic !== "string" || !topic.trim()) {
        res.status(400).json({ error: "Topic is required and must be a non-empty string." });
        return;
      }

      const cleanTopic = topic.trim().slice(0, 100);
      const cleanDifficulty = ["easy", "medium", "hard"].includes(
        String(difficulty).toLowerCase()
      )
        ? String(difficulty).toLowerCase()
        : "medium";

      const prevList = Array.isArray(previousQuestions)
        ? previousQuestions.slice(-20).map((q: unknown) => String(q).trim()).filter(Boolean)
        : [];

      const ai = getGeminiClient();

      const avoidText =
        prevList.length > 0
          ? `\nAlready asked questions in this session (DO NOT repeat or ask very similar variations):\n- ${prevList.join(
              "\n- "
            )}`
          : "";

      const prompt = `You are an expert educational quiz creator. Generate exactly one high-quality multiple choice question about "${cleanTopic}" at "${cleanDifficulty}" difficulty.${avoidText}

Guidelines:
- Create an intriguing, clear question test of knowledge.
- Exactly 4 options that are all plausible, unambiguous, and distinct.
- Exactly one correct answer.
- correctAnswerIndex must be 0, 1, 2, or 3 corresponding to the correct option.
- explanation must be 1 or 2 concise, educational sentences explaining why the correct answer is right.
- difficulty must be "${cleanDifficulty}".
- topic must be "${cleanTopic}".`;

      const candidateModels = ["gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"];
      let responseText: string | undefined;
      let lastError: Error | null = null;

      for (const modelName of candidateModels) {
        try {
          const response = await ai.models.generateContent({
            model: modelName,
            contents: prompt,
            config: {
              temperature: 0.7,
              responseMimeType: "application/json",
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  question: {
                    type: Type.STRING,
                    description: "The quiz question text",
                  },
                  options: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: "Array of exactly four multiple choice answers",
                  },
                  correctAnswerIndex: {
                    type: Type.INTEGER,
                    description: "0-based index of the correct option (0, 1, 2, or 3)",
                  },
                  explanation: {
                    type: Type.STRING,
                    description: "One or two simple sentences explaining the answer",
                  },
                  difficulty: {
                    type: Type.STRING,
                    description: "Difficulty level (easy, medium, hard)",
                  },
                  topic: {
                    type: Type.STRING,
                    description: "The topic of the quiz",
                  },
                },
                required: [
                  "question",
                  "options",
                  "correctAnswerIndex",
                  "explanation",
                  "difficulty",
                  "topic",
                ],
              },
            },
          });

          if (response.text) {
            responseText = response.text;
            break;
          }
        } catch (err) {
          lastError = err instanceof Error ? err : new Error(String(err));
          console.warn(`Model ${modelName} failed with: ${lastError.message}. Trying next candidate if available.`);
        }
      }

      if (!responseText) {
        throw lastError || new Error("No response received from Gemini.");
      }

      let parsedData: QuizQuestion;
      try {
        parsedData = JSON.parse(responseText);
      } catch {
        throw new Error("Failed to parse Gemini response as JSON.");
      }

      // Strict validation
      if (
        !parsedData.question ||
        typeof parsedData.question !== "string" ||
        !parsedData.question.trim()
      ) {
        throw new Error("Invalid question in generated data.");
      }

      if (
        !Array.isArray(parsedData.options) ||
        parsedData.options.length !== 4 ||
        !parsedData.options.every((opt) => typeof opt === "string" && opt.trim().length > 0)
      ) {
        throw new Error("Generated question must have exactly 4 non-empty options.");
      }

      if (
        typeof parsedData.correctAnswerIndex !== "number" ||
        !Number.isInteger(parsedData.correctAnswerIndex) ||
        parsedData.correctAnswerIndex < 0 ||
        parsedData.correctAnswerIndex > 3
      ) {
        throw new Error("correctAnswerIndex must be an integer between 0 and 3.");
      }

      if (
        !parsedData.explanation ||
        typeof parsedData.explanation !== "string" ||
        !parsedData.explanation.trim()
      ) {
        parsedData.explanation = `The correct answer is "${parsedData.options[parsedData.correctAnswerIndex]}".`;
      }

      parsedData.difficulty = cleanDifficulty;
      parsedData.topic = cleanTopic;

      res.json(parsedData);
    } catch (err: unknown) {
      console.error("Error generating question:", err);
      const message = err instanceof Error ? err.message : "Failed to generate question.";
      res.status(500).json({
        error: message,
      });
    }
  });

  // Vite middleware in development; static serve in production
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
});
