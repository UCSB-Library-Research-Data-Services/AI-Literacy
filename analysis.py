"""
AI Literacy LibGuides Analysis Pipeline
=========================================
Comprehensive extraction and synthesis targeting:
1. Definitions of AI Literacy (explicit & implicit)
2. Student Learning Outcomes & Instructor Objectives
3. Guidelines, Policies & Syllabus Recommendations
4. Pedagogical & Instructional Frameworks (e.g., ACRL, ROBOT, CLEAR, Bloom's)
5. Vector Embeddings & Similarity Matching
6. Qualitative Synthesis & Thematic Clustering
"""

import os
import json
import glob
import re
import argparse
from typing import Dict, Any, List, Optional
import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

from config import SERVICES

RESULTS_DIR = "results"
DATA_DIR = "data"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# 1. LLM and Embedding Client Helpers
# ==============================================================================

def call_llm(
    model_identifier: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    response_json: bool = True,
    temperature: float = 0.2,
    timeout: int = 90
) -> str:
    """
    Unified caller supporting:
      - "ollama/<model_name>" (e.g., "ollama/qwen2.5:3b", "ollama/qwen3.6:latest")
      - "dreamlab/<model_name>" (e.g., "dreamlab/gemini-3.7-flash")
      - "grit/<model_name>" (e.g., "grit/llama3.1:8b", "grit/qwen3.5:latest")
      - "aicommons/<model_name>" (e.g., "aicommons/claude-v4.6-sonnet")
    """
    if "/" not in model_identifier:
        provider, model_name = "dreamlab", model_identifier
    else:
        provider, model_name = model_identifier.split("/", 1)
        provider = provider.lower()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if provider == "ollama":
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 32768
            }
        }
        if response_json:
            payload["format"] = "json"

        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    service_map = {
        "dreamlab": "dreamlab",
        "grit": "GRIT",
        "cit": "CIT",
        "aicommons": "AICommons"
    }
    svc_key = service_map.get(provider)
    if not svc_key or svc_key not in SERVICES:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(service_map.keys())} or 'ollama'")

    svc = SERVICES[svc_key]
    url = svc["url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {svc['key']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature
    }
    if response_json:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def get_embedding(
    text: str,
    provider: str = "dreamlab",
    model: str = "gemini-embedding-2"
) -> List[float]:
    """
    Generate vector embeddings locally via Ollama or remotely via DreamLab.
    """
    if provider == "ollama":
        url = "http://localhost:11434/api/embed"
        resp = requests.post(url, json={"model": model, "input": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    elif provider == "dreamlab":
        svc = SERVICES["dreamlab"]
        url = svc["url"].rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {svc['key']}"}
        resp = requests.post(url, headers=headers, json={"model": model, "input": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


# ==============================================================================
# 2. Step 1: Comprehensive Grounded Extraction
# ==============================================================================

EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert qualitative researcher in Information Literacy, Higher Education Pedagogy, and Academic Libraries. "
    "Your objective is to systematically analyze academic library guides (LibGuides) on Artificial Intelligence. "
    "Extract all relevant definitions, objectives, guidelines, outcomes, and instructional frameworks with extreme fidelity to the source text."
)

EXTRACTION_USER_PROMPT_TEMPLATE = """Analyze the LibGuide text for **{institution}**.

Extract all information relevant to AI literacy instruction across five core dimensions:

1. **Definitions of AI Literacy**:
   - `has_explicit_definition`: Boolean (true only if the text formally defines "AI Literacy"; false if it only defines "AI" or discusses it implicitly).
   - `explicit_definition`: The exact or faithful explicit definition if present, or null.
   - `implicit_characterization`: How AI literacy is framed/approached conceptually in this guide.

2. **Objectives & Learning Outcomes**:
   - `student_learning_outcomes`: List of specific skills, capabilities, or knowledge students are expected to acquire (e.g., prompt crafting, evaluating algorithmic bias, hallucination detection).
   - `instructor_pedagogical_goals`: List of recommendations or goals provided for faculty/instructors (e.g., redesigning assignments, syllabus transparency, integrating AI in teaching).

3. **Guidelines & Policies**:
   - `acceptable_vs_prohibited_uses`: Specific rules or guidance on when AI use is permitted or prohibited.
   - `citation_and_attribution_rules`: Guidance on how to document, cite, or disclose AI tool usage (e.g. APA, MLA, Chicago, acknowledgment statements).
   - `syllabus_guidelines`: Guidance for course syllabus policies (e.g. policy tiers, default stances).

4. **Instructional Frameworks & Evaluation Models**:
   - `frameworks`: List of named or structured pedagogical frameworks, heuristics, or evaluation models referenced in the guide (e.g., ACRL Framework for Information Literacy, Bloom's Revised Taxonomy, ROBOT Test, CLEAR prompt framework, CRAAP Test, Leo Lo AI Literacy model, RTF/TAG prompt structures). Each entry must include:
     - `name`: Name of framework.
     - `purpose_and_application`: How the guide applies it to literacy instruction.

5. **Overall Typology & Evidence**:
   - `primary_orientation`: One of ["Tool-Use / Practical Productivity", "Critical & Ethical Evaluation", "Academic Integrity & Citation", "Comprehensive / Balanced Hybrid"].
   - `target_audience`: Stated or primary target audience (e.g., "Students", "Faculty/Instructors", "Researchers", "General Campus").
   - `supporting_quotes`: 3 to 6 verbatim quotes copied word-for-word from the text that directly support the extracted definitions, outcomes, guidelines, or frameworks.

Return ONLY a valid JSON object matching this schema:
{{
  "institution": "{institution}",
  "definitions": {{
    "has_explicit_definition": true,
    "explicit_definition": "...",
    "implicit_characterization": "..."
  }},
  "objectives_and_outcomes": {{
    "student_learning_outcomes": ["..."],
    "instructor_pedagogical_goals": ["..."]
  }},
  "guidelines_and_policies": {{
    "acceptable_vs_prohibited_uses": "...",
    "citation_and_attribution_rules": "...",
    "syllabus_guidelines": "..."
  }},
  "instructional_frameworks": [
    {{
      "name": "...",
      "purpose_and_application": "..."
    }}
  ],
  "primary_orientation": "...",
  "target_audience": "...",
  "supporting_quotes": ["..."]
}}

--- TEXT FOR {institution} ---
{guide_text}
"""


def verify_quotes_in_text(quotes: List[str], raw_text: str) -> Dict[str, Any]:
    """
    Checks if extracted quotes are true verbatim substrings of the original text.
    Computes a fidelity score to flag potential hallucinations.
    """
    cleaned_raw = " ".join(raw_text.split())
    matches = 0
    quote_details = []

    for q in quotes:
        cleaned_q = " ".join(q.split()).strip('\"\'')
        is_exact = cleaned_q in cleaned_raw
        if is_exact:
            matches += 1
        quote_details.append({"quote": q, "found_verbatim": is_exact})

    accuracy = (matches / len(quotes)) if quotes else 1.0
    return {
        "total_quotes": len(quotes),
        "verified_quotes": matches,
        "fidelity_score": round(accuracy, 2),
        "details": quote_details
    }


def clean_json_output(raw_output: str) -> Dict[str, Any]:
    """Strips markdown code fences and parses JSON safely."""
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return json.loads(text.strip())


def run_extraction_for_files(
    model_identifier: str = "dreamlab/gemini-3.7-flash",
    data_dir: str = DATA_DIR,
    force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """
    Extracts structured AI literacy profiles for all text files in data_dir.
    """
    safe_model_tag = model_identifier.replace("/", "_").replace(":", "_")
    output_json = os.path.join(RESULTS_DIR, f"extractions_{safe_model_tag}.json")

    if not force_refresh and os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if isinstance(cached, list) and len(cached) > 0 and all("error" not in item for item in cached):
                    print(f"\n[Step 1] Loaded {len(cached)} existing extractions from cache: {output_json}")
                    return cached
        except Exception:
            pass

    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    print(f"\n[Step 1] Extracting AI literacy profiles for {len(file_paths)} institutions using: {model_identifier}")
    results = []

    for path in file_paths:
        inst_name = os.path.splitext(os.path.basename(path))[0]
        print(f"  -> Analyzing {inst_name}...", end="", flush=True)

        with open(path, "r", encoding="utf-8") as f:
            guide_text = f.read()

        prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            institution=inst_name,
            guide_text=guide_text
        )

        try:
            raw_response = call_llm(
                model_identifier=model_identifier,
                prompt=prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                response_json=True
            )
            extracted = clean_json_output(raw_response)

            # Grounding check
            quotes = extracted.get("supporting_quotes", [])
            verification = verify_quotes_in_text(quotes, guide_text)
            extracted["quote_verification"] = verification
            extracted["raw_file"] = path

            results.append(extracted)
            print(f" Done. (Fidelity: {verification['fidelity_score'] * 100:.0f}%, Frameworks: {len(extracted.get('instructional_frameworks', []))})")
        except Exception as e:
            print(f" Failed: {e}")
            results.append({
                "institution": inst_name,
                "error": str(e),
                "raw_file": path
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Flattened detailed CSV
    summary_rows = []
    for r in results:
        if "error" not in r:
            defs = r.get("definitions", {})
            outcomes = r.get("objectives_and_outcomes", {})
            policies = r.get("guidelines_and_policies", {})
            frameworks = r.get("instructional_frameworks", [])

            fw_names = ", ".join([f.get("name", "") for f in frameworks if f.get("name")])
            student_outcomes = "; ".join(outcomes.get("student_learning_outcomes", []))
            instructor_goals = "; ".join(outcomes.get("instructor_pedagogical_goals", []))

            summary_rows.append({
                "Institution": r.get("institution"),
                "Explicit AI Lit Definition": "Yes" if defs.get("has_explicit_definition") else "No",
                "Primary Orientation": r.get("primary_orientation"),
                "Target Audience": r.get("target_audience"),
                "Instructional Frameworks": fw_names,
                "Student Learning Outcomes": student_outcomes,
                "Instructor Pedagogical Goals": instructor_goals,
                "Acceptable vs Prohibited Uses": policies.get("acceptable_vs_prohibited_uses", ""),
                "Citation & Attribution Rules": policies.get("citation_and_attribution_rules", ""),
                "Syllabus Guidelines": policies.get("syllabus_guidelines", ""),
                "Quote Fidelity": r.get("quote_verification", {}).get("fidelity_score", 0.0),
                "Explicit Definition Text": defs.get("explicit_definition", ""),
                "Implicit Characterization Text": defs.get("implicit_characterization", "")
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(RESULTS_DIR, f"extractions_summary_{safe_model_tag}.csv")
    df_summary.to_csv(summary_csv, index=False)

    print(f"[Step 1 Complete] Saved JSON: {output_json}")
    print(f"[Step 1 Complete] Saved CSV:  {summary_csv}")
    return results


# ==============================================================================
# 3. Step 2: Vector Embeddings & Similarity Matching
# ==============================================================================

def compute_similarity_matrix(
    extractions: List[Dict[str, Any]],
    embed_provider: str = "dreamlab",
    embed_model: str = "gemini-embedding-2",
    plot_filename: str = "cosine_similarity_heatmap.png"
) -> Dict[str, Any]:
    """
    Computes pairwise Cosine Similarity Matrix across institutions incorporating definitions,
    instructional frameworks, student outcomes, and policy guidelines.
    """
    valid_items = [item for item in extractions if "error" not in item]
    if not valid_items:
        raise ValueError("No valid extractions available to compute embeddings.")

    institutions = [item["institution"] for item in valid_items]

    texts_to_embed = []
    for item in valid_items:
        defs = item.get("definitions", {})
        outcomes = item.get("objectives_and_outcomes", {})
        policies = item.get("guidelines_and_policies", {})
        frameworks = item.get("instructional_frameworks", [])

        fw_text = " ".join([f"{f.get('name')}: {f.get('purpose_and_application')}" for f in frameworks])
        outcomes_text = "Student Outcomes: " + ", ".join(outcomes.get("student_learning_outcomes", []))
        instructor_text = "Instructor Goals: " + ", ".join(outcomes.get("instructor_pedagogical_goals", []))
        policy_text = f"Policies: {policies.get('acceptable_vs_prohibited_uses', '')} Citation: {policies.get('citation_and_attribution_rules', '')}"
        
        combined_repr = (
            f"Institution: {item.get('institution')}. "
            f"Orientation: {item.get('primary_orientation', '')}. "
            f"Definition: {defs.get('explicit_definition') or defs.get('implicit_characterization', '')}. "
            f"Frameworks: {fw_text}. "
            f"{outcomes_text}. {instructor_text}. {policy_text}."
        ).strip()
        texts_to_embed.append(combined_repr)

    print(f"\n[Step 2] Generating semantic embeddings for {len(institutions)} institutions using {embed_provider} ({embed_model})...")
    embeddings = []
    for inst, text in zip(institutions, texts_to_embed):
        print(f"  -> Embedding {inst}...", end="", flush=True)
        vec = get_embedding(text, provider=embed_provider, model=embed_model)
        embeddings.append(vec)
        print(" Done.")

    embeddings_matrix = np.array(embeddings)
    sim_matrix = cosine_similarity(embeddings_matrix)

    df_sim = pd.DataFrame(sim_matrix, index=institutions, columns=institutions)

    sim_csv_path = os.path.join(RESULTS_DIR, "cosine_similarity_matrix.csv")
    df_sim.to_csv(sim_csv_path)

    nearest_neighbors = {}
    for inst in institutions:
        series = df_sim.loc[inst].drop(inst).sort_values(ascending=False)
        top_matches = [
            {"institution": neighbor, "similarity": round(float(score), 4)}
            for neighbor, score in series.head(2).items()
        ]
        nearest_neighbors[inst] = top_matches

    # Generate Clustered Heatmap Visualization
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    g = sns.clustermap(
        df_sim,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=df_sim.values[df_sim.values < 0.999].min() * 0.95,
        vmax=1.0,
        linewidths=0.5,
        figsize=(11, 9),
        cbar_kws={'label': 'Cosine Similarity'}
    )
    g.fig.suptitle("AI Literacy Curricular Alignment Across Academic LibGuides", y=1.02, fontsize=14, fontweight="bold")

    plot_path = os.path.join(RESULTS_DIR, plot_filename)
    g.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    print(f"[Step 2 Complete] Heatmap saved to {plot_path}")
    print(f"[Step 2 Complete] Matrix saved to {sim_csv_path}")

    return {
        "dataframe": df_sim,
        "nearest_neighbors": nearest_neighbors,
        "plot_path": plot_path,
        "csv_path": sim_csv_path
    }


# ==============================================================================
# 4. Step 3: Qualitative Thematic Synthesis
# ==============================================================================

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a lead educational researcher and information literacy scholar specializing in academic library instruction and GenAI curricula. "
    "Your task is to synthesize the multi-institutional analysis into a rigorous, publication-grade qualitative research report."
)

SYNTHESIS_USER_PROMPT_TEMPLATE = """You are provided with structured data extracted from {count} university library guides (LibGuides) analyzing their definitions, objectives, guidelines, student outcomes, and instructional frameworks for AI literacy.

### STRUCTURED INSTITUTIONAL DATA:
{extractions_json}

### VECTOR NEAREST NEIGHBOR SIMILARITY:
{neighbors_json}

### INSTRUCTIONS:
Generate a thorough, evidence-grounded Qualitative Research Synthesis in Markdown format with the following core sections:

# Comprehensive Analysis: Frameworks, Guidelines, and AI Literacy Instruction in Higher Education

1. **Executive Summary & Typology Overview**
   - High-level landscape of how academic libraries are defining and operationalizing AI literacy.
   - Summary of institutional distributions across primary orientations (*Balanced Hybrid*, *Pragmatic Tool-Use*, *Critical/Ethical*, *Academic Integrity*).

2. **Definitions of AI Literacy Across Institutions**
   - Analysis of explicit vs. implicit definitions.
   - Prominent conceptual models (e.g., Leo Lo’s multidimensional definition, Bloom-aligned definitions, ACRL adaptation).
   - How libraries define the relationship between "AI Literacy", "Information Literacy", and "Digital Literacy".

3. **Instructional Objectives & Learning Outcomes**
   - **Student Learning Outcomes**: Common competencies (prompt engineering, hallucination/bias verification, attribution, ethics) vs. advanced analytical competencies.
   - **Faculty & Instructor Pedagogical Goals**: How guides instruct educators (assignment redesign, syllabus statements, authentic assessment).

4. **Institutional Guidelines, Policies, and Citation Practices**
   - Acceptable vs. prohibited use standards (spectrum of use models).
   - Citation rules across major style guides (APA, MLA, Chicago) and disclosure requirements.
   - Course syllabus policy recommendations and template tiers.

5. **Instructional Frameworks & Heuristics for Literacy Instruction**
   - Detailed analysis of named frameworks referenced across guides:
     - **ACRL Framework for Information Literacy** (Information Creation as a Process, Authority is Constructed and Contextual, Searching as Strategic Exploration).
     - **Evaluation Rubrics**: The ROBOT Test, CRAAP Test adaptations, Fact-checking heuristics.
     - **Prompting Heuristics**: CLEAR framework, RTF (Role-Task-Format), TAG (Task-Action-Goal).
     - **Cognitive Taxonomies**: Bloom’s Revised Taxonomy adapted for AI workflows.

6. **Institutional Clustering & Curricular Nearest Neighbors**
   - Discussion of the vector similarity findings (which institutions align most closely in pedagogy, and why).
   - Distinctive / outlier institutional approaches (e.g., critical socio-technical focus vs. pure vendor discovery tool focus).

7. **Strategic Recommendations for Academic Library AI Curricula**
   - Actionable roadmap for institutions designing or updating their AI literacy guides and instructional programming.
"""


def run_synthesis(
    extractions: List[Dict[str, Any]],
    nearest_neighbors: Dict[str, Any],
    model_identifier: str = "dreamlab/gemini-3.7-flash"
) -> str:
    """
    Synthesizes the extracted dimensions into a structured research report.
    """
    print(f"\n[Step 3] Running qualitative synthesis with model: {model_identifier}...")

    prompt = SYNTHESIS_USER_PROMPT_TEMPLATE.format(
        count=len(extractions),
        extractions_json=json.dumps(extractions, indent=2),
        neighbors_json=json.dumps(nearest_neighbors, indent=2)
    )

    report_markdown = call_llm(
        model_identifier=model_identifier,
        prompt=prompt,
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        response_json=False,
        temperature=0.3
    )

    report_path = os.path.join(RESULTS_DIR, "qualitative_synthesis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    print(f"[Step 3 Complete] Synthesis report saved to {report_path}")
    return report_markdown


# ==============================================================================
# 5. Multi-Model Comparison Workflow
# ==============================================================================

def compare_models(
    model_identifiers: List[str] = [
        "dreamlab/gemini-3.7-flash",
        "aicommons/claude-v4.6-sonnet",
        "grit/llama3.1:8b"
    ]
) -> pd.DataFrame:
    """
    Runs extraction across multiple models to benchmark reliability, framework capture,
    and verbatim quote fidelity.
    """
    print(f"\n[Comparison Mode] Comparing {len(model_identifiers)} models: {model_identifiers}")
    all_runs = {}

    for model_id in model_identifiers:
        try:
            res = run_extraction_for_files(model_identifier=model_id, force_refresh=True)
            all_runs[model_id] = res
        except Exception as e:
            print(f"Error running model {model_id}: {e}")

    rows = []
    institutions = sorted(list({item["institution"] for res in all_runs.values() for item in res if "institution" in item}))

    for inst in institutions:
        row = {"Institution": inst}
        for model_id, res in all_runs.items():
            inst_data = next((x for x in res if x.get("institution") == inst), {})
            tag = model_id.split("/")[-1]
            defs = inst_data.get("definitions", {})
            fws = inst_data.get("instructional_frameworks", [])
            row[f"{tag}_Orientation"] = inst_data.get("primary_orientation", "N/A")
            row[f"{tag}_ExplicitDef"] = "Yes" if defs.get("has_explicit_definition") else "No"
            row[f"{tag}_FrameworksCount"] = len(fws)
            row[f"{tag}_Fidelity"] = inst_data.get("quote_verification", {}).get("fidelity_score", "N/A")
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    comp_path = os.path.join(RESULTS_DIR, "multi_model_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    print(f"\n[Comparison Complete] Comparison matrix saved to {comp_path}")
    return df_comp


# ==============================================================================
# 6. Main Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="AI Literacy LibGuides Analysis Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="dreamlab/gemini-3.7-flash",
        help="Model identifier (e.g., 'dreamlab/gemini-3.7-flash', 'ollama/qwen2.5:3b', 'grit/llama3.1:8b', 'aicommons/claude-v4.6-sonnet')"
    )
    parser.add_argument(
        "--embed-provider",
        type=str,
        default="dreamlab",
        choices=["ollama", "dreamlab"],
        help="Embedding provider ('dreamlab' for gemini-embedding-2, 'ollama' for nomic-embed-text)"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="gemini-embedding-2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run multi-model comparison mode across DreamLab, AICommons, and GRIT/Ollama"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running extractions even if cached results exist"
    )

    args = parser.parse_args()

    if args.compare:
        compare_models([
            "dreamlab/gemini-3.7-flash",
            "aicommons/claude-v4.6-sonnet",
            "grit/llama3.1:8b"
        ])
        return

    # 1. Extraction (with force refresh option)
    extractions = run_extraction_for_files(model_identifier=args.model, force_refresh=args.force)

    # 2. Embeddings & Cosine Similarity
    sim_results = compute_similarity_matrix(
        extractions=extractions,
        embed_provider=args.embed_provider,
        embed_model=args.embed_model
    )

    # 3. Qualitative Synthesis Report
    synthesis_report = run_synthesis(
        extractions=extractions,
        nearest_neighbors=sim_results["nearest_neighbors"],
        model_identifier=args.model
    )

    print("\n=======================================================")
    print("Pipeline completed successfully!")
    print(f"1. Extractions: results/extractions_*.json")
    print(f"2. Summary CSV: results/extractions_summary_*.csv")
    print(f"3. Heatmap:     {sim_results['plot_path']}")
    print(f"4. Matrix CSV:  {sim_results['csv_path']}")
    print(f"5. Synthesis:   results/qualitative_synthesis_report.md")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
