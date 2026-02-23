import json
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

class Transformer:
    def __init__(self):
        # ⚠️ IMPORTANT: Ensure you have the spaCy model installed before running this code.
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            print("⚠️  spaCy model 'en_core_web_lg' not found. Please install it with: python -m spacy download en_core_web_lg")
            raise

        self.base_path = Path(__file__).parent.parent.parent
        self.raw_path = self.base_path / "data" / "raw"
        self.proc_path = self.base_path / "data" / "processed"
        self.proc_path.mkdir(parents=True, exist_ok=True)

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        # 🚀 EXPANDED TAXONOMY (Adding more common tech)
        base_skills = [
            "Python", "JavaScript", "TypeScript", "Node.js", "React", "Vue", 
            "Java", "C#", "C++", "Go", "Rust", "PHP", "Ruby", "SQL", "NoSQL",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "LLM", "AI", "ML"
        ]
        patterns = [self.nlp.make_doc(text) for text in base_skills]
        self.matcher.add("KNOWN_TECH", patterns)

    def transform(self):
        # Find all JSON files in the raw data directory
        json_files = list(self.raw_path.rglob("year=*/month=*/*.json"))
        if not json_files:
            print("❌ No JSON files found in raw data path.")
            return

        # Process each file and extract skills
        for file in json_files:
            # Maintain the same directory structure in processed folder
            relative_path = file.relative_to(self.raw_path).parent
            output_dir = self.proc_path / relative_path
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"🧠 NLP Processing: {file.name} -> {relative_path}")

            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            posts = data.get('posts', [])
            results = []

            for post in posts:
                # Clean HTML
                raw_text = post.get('text', '')
                clean_text = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ")
                doc = self.nlp(clean_text)

                # 1. Look for our known skills
                matches = self.matcher(doc)
                found = set([doc[start:end].text.title() for _, start, end in matches])

                # 2. ADVANCED: Look for "Product" entities we missed
                # This finds things like "Snowflake" or "Redis" even if not in our list
                for ent in doc.ents:
                    if ent.label_ in ["PRODUCT", "ORG", "WORK_OF_ART"]:
                        # Simple logic: if it's one word and capitalized, it might be a tech
                        if len(ent.text.split()) == 1 and ent.text[0].isupper():
                            found.add(ent.text)

                results.append({
                    'id': post.get('id'),
                    'month': file.name.split('_')[0],
                    'skills': "|".join(sorted(list(found))) if found else "Unclassified",
                    'has_skills': 1 if found else 0,
                    'full_text': clean_text[:200] # Keeping a snippet for verification
                })

            df = pd.DataFrame(results)
            current_date = datetime.now().strftime("%Y-%m")
            output_file = output_dir / f"{current_date}_NLP_extracted.csv"
            df.to_csv(output_file, index=False)
            
            # Print quality report
            coverage = (df['has_skills'].sum() / len(df)) * 100
            print(f"   ✅ Saved: {output_file} (Coverage: {coverage:.1f}%)")

if __name__ == "__main__":
    Transformer().transform()