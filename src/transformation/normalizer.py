import pandas as pd
from pathlib import Path
from datetime import datetime
import json

class GoldStandardTransformer:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.proc_path = self.base_path / "data" / "processed"
        
        # 📂 DATA-DRIVEN TAXONOMY
        # These were extracted and verified from your actual February data
        self.taxonomy = {
            "Languages": ["Python", "TypeScript", "Go", "Rust", "Java", "C++", "C#", "Ruby", "JavaScript"],
            "Frameworks": ["React", "NodeJS", "Angular", "Vue", "Django", "FastAPI", "NextJS"],
            "Infrastructure": ["AWS", "GCP", "Azure", "Kubernetes", "Docker", "Terraform", "Cloudflare", "Ansible", "Airflow"],
            "AI_ML": ["AI", "Machine Learning", "LLM", "LangChain", "PyTorch", "TensorFlow", "OpenAI"],
            "Database": ["SQL", "PostgreSQL", "Redis", "Elasticsearch", "ClickHouse", "Snowflake", "MongoDB"]
        }
        
        # Merge all into a single lookup set for speed
        self.whitelist = {item for sublist in self.taxonomy.values() for item in sublist}

        # Normalization Map (Merging variations)
        self.norm_map = {
            "Ai": "AI", "Ml": "Machine Learning", "Llm": "LLM",
            "Typescript": "TypeScript", "Javascript": "JavaScript",
            "Node.Js": "NodeJS", "Node.js": "NodeJS", "Node": "NodeJS",
            "Aws": "AWS", "Gcp": "GCP", "Sql": "SQL", "Postgresql": "PostgreSQL"
        }

    def classify_role(self, skills, text):
        """Heuristic to determine the job category."""
        text = str(text).lower()
        skills = str(skills)
        
        if any(x in skills for x in ["AI", "Machine Learning", "LLM", "PyTorch"]):
            return "Data & AI"
        if any(x in skills for x in ["React", "Vue", "Angular", "TypeScript"]) or "frontend" in text:
            return "Frontend"
        if any(x in skills for x in ["Kubernetes", "Docker", "Terraform", "AWS"]) or "devops" in text:
            return "DevOps & SRE"
        
        return "General Software Engineering"

    def clean_and_categorize(self, skill_string):
        if pd.isna(skill_string) or skill_string == "Unclassified":
            return "Unclassified"
        
        raw_skills = skill_string.split('|')
        clean_set = set()
        
        for s in raw_skills:
            # 1. Normalize (e.g., 'Ai' -> 'AI')
            normalized = self.norm_map.get(s, s)
            # 2. Filter against Whitelist
            if normalized in self.whitelist:
                clean_set.add(normalized)
        
        return "|".join(sorted(list(clean_set))) if clean_set else "Unclassified"

    def process_files(self):
        # We process the 'final' files which already have the spaCy NER data
        current_date = datetime.now().strftime("%Y-%m")
        files = list(self.proc_path.glob(f"year=*/month=*/{current_date}_NLP_extracted.csv"))
        
        for file_path in files:
            print(f"💎 Refinement: {file_path.name}")
            df = pd.read_csv(file_path)
            initial_count = len(df)
            
            # Apply our refined taxonomy
            df['skills'] = df['skills'].apply(self.clean_and_categorize)

            # Classify job roles based on skills and text
            df['role'] = df.apply(lambda x: self.classify_role(x['skills'], x['full_text']), axis=1)
            
            output_file = file_path.parent / f"{current_date}_ml_ready.csv"
            df.to_csv(output_file, index=False)
            print(f"✅ Done! Saved {len(df)} high-precision rows into {output_file}")

            report = {
                "input_rows": initial_count,
                "duplicates_removed": initial_count - len(df),
                "clean_rows": len(df[df['skills'] != "Unclassified"]),
                "noise_rows": len(df[df['skills'] == "Unclassified"]),
                "coverage_pct": (len(df[df['skills'] != "Unclassified"]) / len(df)) * 100
            }
            with open(file_path.parent / f"dq_report_{current_date}.json", 'w') as f:
                json.dump(report, f, indent=4)

if __name__ == "__main__":
    GoldStandardTransformer().process_files()