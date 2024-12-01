import { Agent, Prompt, InitI18n } from "agent-builder";
import { OpenAIModel } from "agent-builder/lib/models";
import { JSONSchemaParser } from "agent-builder/lib/parsers";

import fs from "fs";
import dotenv from "dotenv";

dotenv.config();

async function main() {
    const dataset = loadDataset("movie_review_dataset.json");

    InitI18n({
        lng: "pt",
    });

    const model = new OpenAIModel({
        model: "gpt-4o-mini",
        apiKey: process.env.OPENAI_API_KEY,
    });

    type Review = {
        avaliacao: "positivo" | "negativo";
        tags: string[];
    };

    const parser = new JSONSchemaParser<Review>({
        type: "object",
        properties: {
            avaliacao: {
                type: "string",
                enum: ["positivo", "negativo"],
            },
            tags: {
                type: "array",
                items: { type: "string" },
            },
        },
        required: ["avaliacao", "tags"],
    });

    const agent = new Agent<Review>(model, parser);

    const prompt = new Prompt([
        {
            role: "system",
            content: "Você é um assistente de avaliação de filmes. \
            Você irá receber uma avaliação de um filme e irá classificar \
            se a avaliação é positiva ou negativa e gerar tags sobre a avaliação. ",
        },
        { role: "user", content: "Avalie a seguinte avaliação: {{review}}" },
    ]);

    const results = [];
    for (const review of dataset) {
        const response = await agent.execute(prompt.format({ review }));
        results.push(response.data);
    }

    fs.writeFileSync("results_agent.json", JSON.stringify(results, null, 2));
}

function loadDataset(filePath: string) {
    const data = fs.readFileSync(filePath, "utf-8");
    const dataset = JSON.parse(data);
    return [...dataset["positive"].slice(0, 100), ...dataset["negative"].slice(0, 100)];
}

main();
