import json
import pandas as pd
import spacy
import logging
from spacy.matcher import PhraseMatcher
from bs4 import BeautifulSoup
from pathlib import Path
from esco_engine import ESCOTaxonomyEngine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Transformer:
    def __init__(self):
        # ⚠️ IMPORTANT: Ensure you have the spaCy model installed before running this code.
        try:
            # Keep only NER for speed
            self.nlp = spacy.load(
                "en_core_web_lg",
                disable=["parser", "tagger", "lemmatizer", "attribute_ruler"]
            )
            logger.info("✅ spaCy model 'en_core_web_lg' loaded successfully")
        except OSError:
            logger.error("⚠️  spaCy model 'en_core_web_lg' not found. Please install it with: python -m spacy download en_core_web_lg")
            raise

        self.base_path = Path(__file__).parent.parent.parent
        self.raw_path = self.base_path / "data" / "raw"
        self.proc_path = self.base_path / "data" / "processed"
        self.proc_path.mkdir(parents=True, exist_ok=True)

        # Performance tuning
        self.max_text_len = 1000
        self.snippet_len = 200
        self.batch_size = 64
        self.n_process = 1

        # Initialize the New ESCO Engine
        self.esco = ESCOTaxonomyEngine(
            skills_csv=self.base_path / "data/external/esco/skills_en.csv",
            digital_filter_csv=self.base_path / "data/external/esco/DigitalSkill_en.csv"
        )
        logger.info("✅ ESCO Engine initialized")

        self.base_skills = [
            # Languages/Runtime
            "Python", "JavaScript", "TypeScript", "Node.js", "Java", "C#", "C++", "Go", "Rust",
            "PHP", "Ruby", "SQL", "NoSQL", "Bash", "PowerShell", "Scala", "Kotlin", "Swift",
            "R", "MATLAB", "Julia",

            # Frontend
            "React", "Vue", "Angular", "Svelte", "Next.js", "Nuxt", "React Native",
            "Redux", "Tailwind", "Webpack", "Vite",

            # Backend
            "FastAPI", "Django", "Flask", "Spring", "Spring Boot", "Express", "NestJS",
            ".NET", "ASP.NET", "gRPC",

            # Data/ML
            "AI", "ML", "LLM", "PyTorch", "TensorFlow", "pandas", "NumPy",
            "scikit-learn", "XGBoost", "LightGBM", "CatBoost", "MLflow",
            "Hugging Face", "LangChain", "spaCy",

            # Data Engineering
            "Kafka", "Spark", "Hadoop", "Airflow", "dbt", "Databricks",
            "Flink", "Beam", "Trino", "Presto",

            # Databases
            "PostgreSQL", "MySQL", "MongoDB", "DynamoDB", "Cassandra",
            "Neo4j", "Elasticsearch", "Redis", "Snowflake",

            # Streaming/Messaging
            "RabbitMQ", "Kinesis", "Pub/Sub", "ActiveMQ", "NATS",

            # Cloud/DevOps
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "CI/CD",
            "GitHub Actions", "GitLab CI", "Jenkins", "Argo CD", "Helm",
            "Ansible", "Terraform", "Prometheus", "Grafana", "Sentry",

            # Security/Identity
            "OAuth", "OIDC", "JWT", "SAML", "Vault",

            # Data Warehouse/BI
            "BigQuery", "Redshift", "Synapse", "Looker", "Tableau", "Power BI",

            # Platforms/Other
            "Terraform", "Snowflake", "Redis"
        ]

        # Deduplicate while preserving order
        seen = set()
        self.base_skills = [s for s in self.base_skills if not (s in seen or seen.add(s))]
        logger.info(f"✅ Loaded {len(self.base_skills)} base skills")

        # Prebuild PhraseMatcher for fast skill detection
        self.skill_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(skill) for skill in self.base_skills]
        self.skill_matcher.add("BASE_SKILLS", patterns)
        logger.info("✅ PhraseMatcher initialized")

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        return BeautifulSoup(raw_text or "", "html.parser").get_text(separator=" ")

    def _extract_skills(self, text: str) -> set:
        found = set(self.esco.extract(text))
        doc = self.nlp.make_doc(text)
        for _, start, end in self.skill_matcher(doc):
            found.add(doc[start:end].text)
        return found

    def transform(self):
        # Find all JSON files in the raw data directory
        json_files = list(self.raw_path.rglob("year=*/month=*/*.json"))
        if not json_files:
            logger.error("❌ No JSON files found in raw data path.")
            return

        logger.info(f"📂 Found {len(json_files)} JSON files to process")

        # Process each file and extract skills
        for file_idx, file in enumerate(json_files, 1):
            try:
                # Maintain the same directory structure in processed folder
                relative_path = file.relative_to(self.raw_path).parent
                output_dir = self.proc_path / relative_path
                output_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"🧠 [{file_idx}/{len(json_files)}] Processing: {file.name} -> {relative_path}")

                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                posts = data.get('posts', [])
                if not posts:
                    logger.warning(f"⚠️  No posts found in {file.name}")
                    continue

                results = []

                # Pre-clean and pre-trim text
                cleaned_texts = [
                    self._clean_text(p.get('text', ''))[: self.max_text_len]
                    for p in posts
                ]

                # Batch NER for speed
                docs = self.nlp.pipe(cleaned_texts, batch_size=self.batch_size, n_process=self.n_process)

                month = file.name.split('_')[0]

                for post, clean_text, doc in zip(posts, cleaned_texts, docs):
                    found = self._extract_skills(clean_text)

                    # ADVANCED: Look for "Product" entities we missed
                    for ent in doc.ents:
                        if ent.label_ in ["PRODUCT", "WORK_OF_ART"] and len(ent.text) > 2:
                            if len(ent.text.split()) == 1 and ent.text[0].isupper():
                                found.add(ent.text.strip())

                    results.append({
                        'id': post.get('id'),
                        'month': month,
                        'skills': "|".join(sorted(list(found))) if found else "Unclassified",
                        'has_skills': 1 if found else 0,
                        'full_text': clean_text[: self.snippet_len]  # snippet for verification
                    })

                df = pd.DataFrame(results)
                output_file = output_dir / "NLP_extracted.csv"
                df.to_csv(output_file, index=False)

                # Print quality report
                coverage = (df['has_skills'].sum() / len(df)) * 100 if len(df) else 0
                logger.info(f"   ✅ Saved: {output_file} (Coverage: {coverage:.1f}%)")

            except Exception as e:
                logger.error(f"❌ Error processing {file.name}: {e}", exc_info=True)
                continue

        logger.info("🎉 Transformation complete!")

if __name__ == "__main__":
    Transformer().transform()