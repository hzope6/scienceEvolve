from google import genai
import os
import json


from google import genai
import os
import json

# --- AUTH ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# --- LOAD FILES ---
with open("question_data.json", "r") as f:
    question_data = json.load(f)

with open("prompt_for_LLM_Judge.txt", "r") as f:
    judge_prompt = f.read().strip()

# --- PERSONAS ---
physicist_persona = """You are a physicist trained to explain the world through underlying laws, conservation principles, and quantitative reasoning. 
You prefer mechanistic, reductionist explanations and test hypotheses using modeling, mathematics, or thought experiments.
At each step, form a hypothesis, test it conceptually, revise, and synthesize a unified model."""

econ_persona = """You are an economist who analyzes systems through incentives, feedback loops, and equilibrium constraints. 
You reason about agents, markets, and institutions using cause-and-effect logic, and you test hypotheses by tracing behavioral and macroeconomic responses.
At each step, form a hypothesis about incentives or constraints, test it against expected outcomes, revise, and synthesize an equilibrium explanation."""

# call gemini to get the judge scores fro the question_data[" "example_reasoning_path""] which is a list of strings. 
# print out the string and the scores.



# --- LLM Judge Function ---
def get_judge_score(question, ground_truth, reasoning_steps, persona):
    """Call Gemini to evaluate how close the reasoning is to the ground truth."""
    reasoning_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(reasoning_steps)])
    prompt = f"""{judge_prompt}

Persona:
{persona}

Question:
{question}

Ground Truth:
{ground_truth}

Reasoning Path:
{reasoning_text}

Please output a JSON with fields:
- score (float between 0 and 1)
- explanation (short rationale for the score)
"""
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[{"role": "user", "parts": [prompt]}],
    )
    return response.text

# --- MAIN LOOP ---
for ex in question_data[:2]:  # test a few first
    print(f"\n=== Question {ex['id']} ===")
    for i, reasoning_path in enumerate(example_reasoning_paths):
        print(f"\n--- Path {i+1} ---")
        result = get_judge_score(
            question=ex["question"],
            ground_truth=ex["ground_truth"],
            reasoning_steps=reasoning_path,
            persona=physicist_persona
        )
        print(result)
