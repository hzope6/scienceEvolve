#!/usr/bin/env python3
"""
Self-Evolving Science Agent System
Iteratively improves agent reasoning through LLM-based evaluation and playbook evolution.
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from google import genai

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Please set it in .env file or as an environment variable.")

MODEL_NAME = "gemini-2.0-flash-lite"
client = genai.Client(api_key=GOOGLE_API_KEY)

# --- Load Prompts ---
def load_prompt_file(filename: str) -> str:
    """Load a prompt file and return its contents."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"ERROR: Prompt file '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load '{filename}': {e}")
        sys.exit(1)

agent_prompt = load_prompt_file("promp_for agent.txt")
judge_prompt = load_prompt_file("prompt_for_LLM_Judge.txt")
playbook_prompt = load_prompt_file("play_book.txt")

# --- LLM Functions ---
def call_llm(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini API with retry logic."""
    for attempt in range(max_retries):
        try:
            # Use the correct API format: pass prompt as string directly
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"ERROR: LLM API call failed after {max_retries} attempts: {e}")
                raise
            print(f"WARNING: LLM API call failed (attempt {attempt + 1}/{max_retries}), retrying...")
            continue
    return ""

def parse_json_response(response: str, expected_keys: List[str]) -> Dict[str, Any]:
    """Parse JSON response from LLM, handling markdown code blocks if present."""
    # Remove markdown code blocks if present
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    
    try:
        data = json.loads(response)
        # Validate expected keys
        for key in expected_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in response")
        return data
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON response: {e}")
        print(f"Response content: {response[:500]}")
        raise

def solve_problem(persona: str, current_playbook: str, past_records: List[Dict], 
                  current_problem: str) -> Dict[str, str]:
    """Agent solves a problem using current playbook and past records."""
    input_data = {
        "persona": persona,
        "current_playbook": current_playbook,
        "past_records": past_records,
        "current_problem": current_problem
    }
    
    prompt = f"""{agent_prompt}

Input JSON:
{json.dumps(input_data, indent=2)}

Please provide your reasoning process and final answer in the specified JSON format."""
    
    print("  [Agent] Solving problem...")
    response = call_llm(prompt)
    result = parse_json_response(response, ["reasoning_process", "final_answer"])
    return result

def evaluate_solution(reasoning_process: str, final_answer: str, 
                     question: str, ground_truth: str) -> Dict[str, Any]:
    """LLM Judge evaluates the agent's solution."""
    prompt = f"""{judge_prompt}

Question:
{question}

Agent's Reasoning Process:
{reasoning_process}

Agent's Final Answer:
{final_answer}

Ground Truth Answer:
{ground_truth}

Please evaluate and provide your score and feedback in the specified JSON format."""
    
    print("  [Judge] Evaluating solution...")
    response = call_llm(prompt)
    result = parse_json_response(response, ["score", "feedback"])
    
    # Ensure score is a float
    if isinstance(result["score"], str):
        try:
            result["score"] = float(result["score"])
        except ValueError:
            print(f"WARNING: Could not convert score '{result['score']}' to float, using 0.0")
            result["score"] = 0.0
    elif not isinstance(result["score"], (int, float)):
        result["score"] = 0.0
    
    result["score"] = float(result["score"])
    return result

def generate_playbook(persona: str, past_records: List[Dict], 
                     current_problem: str) -> str:
    """Generate a new playbook based on past records and feedback."""
    input_data = {
        "persona": persona,
        "past_records": past_records,
        "current_problem": current_problem
    }
    
    prompt = f"""{playbook_prompt}

Input JSON:
{json.dumps(input_data, indent=2)}

Please generate a new playbook in the specified JSON format."""
    
    print("  [Playbook Generator] Creating new playbook...")
    response = call_llm(prompt)
    result = parse_json_response(response, ["new_playbook"])
    return result["new_playbook"]

# --- File Operations ---
def load_initial_config() -> Dict[str, Any]:
    """Load initial configuration from initial.json."""
    try:
        with open("initial.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ["persona", "initial_playbook", "score_threshold", "question_id"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in initial.json")
        
        return config
    except FileNotFoundError:
        print("ERROR: initial.json not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in initial.json: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load initial.json: {e}")
        sys.exit(1)

def load_physics_question(question_id: int) -> Dict[str, Any]:
    """Load a physics question by ID from question_physics.json."""
    try:
        with open("question_physics.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        for q in questions:
            if q.get("id") == question_id:
                return q
        
        raise ValueError(f"Question with ID {question_id} not found in question_physics.json")
    except FileNotFoundError:
        print("ERROR: question_physics.json not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in question_physics.json: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load question: {e}")
        sys.exit(1)

def save_history(history: List[Dict], filename: str = "history.json"):
    """Save iteration history to JSON file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"  [System] History saved to {filename}")
    except Exception as e:
        print(f"WARNING: Failed to save history to {filename}: {e}")

def load_history(filename: str = "history.json") -> List[Dict]:
    """Load existing history from JSON file if it exists."""
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load existing history: {e}")
            return []
    return []

# --- Main Iteration Loop ---
def main():
    """Main execution function."""
    print("=" * 80)
    print("Self-Evolving Science Agent System")
    print("=" * 80)
    print()
    
    # Load configuration
    print("[Initialization] Loading configuration...")
    config = load_initial_config()
    persona = config["persona"]
    current_playbook = config["initial_playbook"]
    score_threshold = float(config["score_threshold"])
    question_id = int(config["question_id"])
    
    print(f"  Persona: {persona[:80]}...")
    print(f"  Initial Playbook: {current_playbook[:80]}...")
    print(f"  Score Threshold: {score_threshold}")
    print(f"  Question ID: {question_id}")
    print()
    
    # Load question
    print("[Initialization] Loading physics question...")
    question_data = load_physics_question(question_id)
    question = question_data["question"]
    ground_truth = question_data["ground_truth"]
    
    print(f"  Question: {question}")
    print(f"  Ground Truth: {ground_truth}")
    print()
    
    # Initialize history
    past_records: List[Dict] = []
    iteration = 0
    
    print("=" * 80)
    print("Starting Iteration Loop")
    print("=" * 80)
    print()
    
    # Main iteration loop
    while True:
        iteration += 1
        print("-" * 80)
        print(f"ITERATION {iteration}")
        print("-" * 80)
        print()
        
        print(f"[Iteration {iteration}] Current Playbook:")
        print(f"  {current_playbook}")
        print()
        
        # Step 1: Agent solves the problem
        print(f"[Iteration {iteration}] Step 1: Agent solving problem...")
        try:
            solution = solve_problem(
                persona=persona,
                current_playbook=current_playbook,
                past_records=past_records,
                current_problem=question
            )
            
            reasoning_process = solution["reasoning_process"]
            final_answer = solution["final_answer"]
            
            print(f"  Reasoning Process:")
            print(f"  {reasoning_process[:200]}..." if len(reasoning_process) > 200 else f"  {reasoning_process}")
            print()
            print(f"  Final Answer: {final_answer}")
            print()
        except Exception as e:
            print(f"ERROR: Failed to solve problem: {e}")
            sys.exit(1)
        
        # Step 2: Judge evaluates the solution
        print(f"[Iteration {iteration}] Step 2: Judge evaluating solution...")
        try:
            evaluation = evaluate_solution(
                reasoning_process=reasoning_process,
                final_answer=final_answer,
                question=question,
                ground_truth=ground_truth
            )
            
            score = evaluation["score"]
            feedback = evaluation["feedback"]
            
            print(f"  Score: {score:.3f}")
            print(f"  Feedback: {feedback}")
            print()
        except Exception as e:
            print(f"ERROR: Failed to evaluate solution: {e}")
            sys.exit(1)
        
        # Create record for this iteration
        record = {
            "iteration": iteration,
            "reasoning_process": reasoning_process,
            "final_answer": final_answer,
            "feedback": feedback,
            "score": score,
            "previous_playbook": current_playbook
        }
        past_records.append(record)
        
        # Save history after each iteration
        save_history(past_records)
        print()
        
        # Check if threshold is reached
        if score >= score_threshold:
            print("=" * 80)
            print(f"SUCCESS: Score {score:.3f} >= Threshold {score_threshold}")
            print("=" * 80)
            print()
            print(f"Final Playbook:")
            print(f"  {current_playbook}")
            print()
            print(f"Total Iterations: {iteration}")
            print(f"Final Score: {score:.3f}")
            break
        
        # Step 3: Generate new playbook based on feedback
        print(f"[Iteration {iteration}] Step 3: Generating new playbook...")
        try:
            new_playbook = generate_playbook(
                persona=persona,
                past_records=past_records,
                current_problem=question
            )
            
            print(f"  New Playbook: {new_playbook[:200]}..." if len(new_playbook) > 200 else f"  New Playbook: {new_playbook}")
            print()
            
            current_playbook = new_playbook
        except Exception as e:
            print(f"ERROR: Failed to generate playbook: {e}")
            sys.exit(1)
        
        print(f"[Iteration {iteration}] Continuing to next iteration...")
        print()
    
    print("=" * 80)
    print("Execution Complete")
    print("=" * 80)
    print(f"History saved to: history.json")
    print(f"Total iterations: {iteration}")
    print(f"Final score: {score:.3f}")

if __name__ == "__main__":
    main()

