"""
File: src/report/generate_detailed_report.py
Purpose: Transformation utility that converts raw JSONL pipeline results into a human-readable 
Markdown format. It provides a granular, case-by-case breakdown of each evaluation case, 
showing the input evidence, the initial draft answer, and the final gated decision. This 
allows researchers to audit individual successes and failures beyond aggregate metrics, 
enabling qualitative analysis of model reasoning and verification gaps.
"""

import json
from pathlib import Path

def generate_detailed_report(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: {input_path} not found.")
        return

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Detailed Results Breakdown (Groq Evaluation)\n\n")
        out.write("This report provides a case-by-case look at the Verity-H pipeline performance.\n\n")
        
        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    case_id = data.get("case_id", "N/A")
                    category = data.get("category", "N/A")
                    question = data.get("question", "N/A")
                    draft = data.get("draft_answer", "N/A")
                    gate = data.get("gate_output", {})
                    decision = gate.get("decision", "ERROR")
                    final_answer = gate.get("final_answer", "N/A")
                    
                    out.write(f"## [{case_id}] {category.upper()}\n\n")
                    out.write(f"**Question:** {question}\n\n")
                    out.write(f"**Gate Decision:** `{decision}`\n\n")
                    
                    out.write("### Draft Answer (Initial AI response)\n")
                    out.write(f"> {draft}\n\n")
                    
                    out.write("### Final Verity Answer\n")
                    out.write(f"```text\n{final_answer}\n```\n\n")
                    
                    # Add specific notes if it's a contradiction or hypothesis
                    if decision == "contradiction":
                        out.write("#### ⚠️ Contradiction Detected\n")
                        claims = data.get("verifier_output", {}).get("claims", [])
                        for c in claims:
                            if c.get("label") == "CONTRADICTION":
                                out.write(f"- {c.get('claim_text')}\n")
                        out.write("\n")
                    
                    out.write("---\n\n")
                except Exception as e:
                    out.write(f"### Error parsing case\n`{str(e)}`\n\n---\n\n")

    print(f"Detailed report generated: {output_path}")

if __name__ == "__main__":
    generate_detailed_report("results/verity_pipeline_groq.jsonl", "results/GROQ_DETAILED_BREAKDOWN.md")
