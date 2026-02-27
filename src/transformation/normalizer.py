import pandas as pd
import json
import re
from pathlib import Path
from collections import Counter

class Normalizer:
    """Cleans, normalizes, and categorizes tech skills from job postings."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.proc_path = self.base_path / "data" / "processed"
        
        self._init_blacklist()
        self._init_norm_map()
        self._init_skill_categories()

    def _init_blacklist(self):
        """Initialize the blacklist of generic/irrelevant terms."""
        self.blacklist = {
            # Roles / Titles
            'cto', 'ceo', 'coo', 'vp', 'founding engineer', 'founder', 'phd',
            
            # Generic Tech Terms
            'devops', 'backend', 'frontend', 'fullstack', 'full stack', 'ai/ml', 
            'oss', 'e2e', 'etl', 'api', 'sdk', 'mvp', 'dns', 'hpc', 'gpu', 'mcp',
            'oauth', 'infrastructure', 'ci/cd', 'version control', 'microservices',
            'monolith', 'serverless', 'edge computing',
            
            # Broad Domains / Concepts
            'robotics', 'computer vision', 'data science', 'computer science',
            'data models', 'sensors', 'systems thinking', 'smart contract', 'fhir',
            'electronic communication', 'computer programming', 'computer technology',
            'geographic information systems', 'learning management systems', 'database',
            'integrated development environment software', 'augmented reality',
            'outsourcing model', 'business intelligence', 'system design', 'web services',
            'information technology', 'software development', 'firmware', 'biology',
            'hardware testing methods', 'service-oriented modelling', 'computational biology',
            'computer graphics', 'unstructured data', 'hardware components',
            'building automation', "Computer Assisted Language Learning",
            
            # Business / Process Terms
            'saas', 'paas', 'iaas', 'innovation', 'agile', 'scrum', 'waterfall',
            'rpa', 'sla', 'kpi', 'poc', 'b2b', 'b2c', 'arr', 'gtv', 'gtm',
            'cdp', 'erp', 'cms', 'crm', 'iot', 'ats',
            
            # Industries
            'biotech', 'healthcare', 'fintech', 'blockchain', 'crypto', 'finance',
            'tech', 'electronics', 'economics', 'legal', 'compliance', 'investing',
            'trading',
            
            # HR / Recruiting / Employment
            'recruiting', 'talent acquisition', 'hr', 'human resources', 'marketing',
            'sales', 'customer service', 'support', 'pto', 'onsite', 'remote', 'visa',
            
            # Companies
            'zillow', 'brex', 'sequoia', 'microsoft', 'google', 'apple', 'github',
            'accel', 'blackbird', 'dynasty', 'futo', 'rover', 'kredit', 'casus',
            'phonely', 'instagram', 'meta', 'amazon', 'milaboratories', 'spacex',
            'adobe', 'tiktok', 'bloomberg', 'cerner', 'epic', 'atrium', 'repspark',
            'discord', 'slack', 'zoom', 'nvidia', 'starbridge', 'atomscale',
            'immunera', 'comper', 'peregrine', 'onja', 'stripe', 'shopify',
            'salesforce', 'oracle', 'sap', 'workday', 'databricks', 'confluent',
            'vercel', 'netlify', 'heroku',
            
            # Universities / Education
            'mit', 'stanford', 'harvard', 'oxford', 'cambridge', 'iiit-hyderabad',
            'bachelor', 'master', 'degree', 'certification', 'bootcamp',
            
            # Geographies / Regions / Country-specific
            'united', 'aadhaar', 'upi',
            
            # Misc / Noise / Ambiguous
            'source', 'hackernews', 'vlc', 'quic', 'kyber', 'rage', 'cet', 'less',
            'linear', 'algorithms', 'logic', 'cursor', 'ngs',
            'create prototype of user experience solutions', 'Lidar'
        }

    def _init_norm_map(self):
        """Initialize skill normalization mappings by category."""
        languages = {
            "python": "Python", "javascript": "JavaScript", "typescript": "TypeScript",
            "golang": "Go", "go": "Go", "rust": "Rust", "c++": "C++", "c#": "C#",
            "csharp": "C#", "objective-c": "Objective-C", "r": "R", "swift": "Swift",
            "java": "Java", "php": "PHP", "ruby": "Ruby",
        }

        web_tech = {
            "html": "HTML", "css": "CSS", "style sheet languages": "CSS",
            "style sheets": "CSS",
        }

        frontend = {
            "react": "React", "reactjs": "React", "react.js": "React",
            "vue": "Vue", "vuejs": "Vue", "vue.js": "Vue",
            "angular": "Angular", "angularjs": "Angular",
            "next.js": "NextJS", "nextjs": "NextJS", "svelte": "Svelte",
        }

        backend = {
            "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
            "express": "Express", "expressjs": "Express", "django": "Django",
            "flask": "Flask", "fastapi": "FastAPI", "spring boot": "Spring Boot",
            "spring": "Spring", ".net": ".NET", "dotnet": ".NET",
            "rails": "Ruby on Rails", "laravel": "Laravel",
        }

        cloud = {
            "aws": "AWS", "amazon web services": "AWS", "amazon": "AWS", "ec2": "AWS",
            "s3": "AWS", "gcp": "GCP", "google cloud": "GCP",
            "google cloud platform": "GCP", "azure": "Azure", "microsoft azure": "Azure",
        }

        databases = {
            "sql": "SQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
            "mysql": "MySQL", "sqlite": "SQLite", "mongodb": "MongoDB", "mongo": "MongoDB",
            "dynamodb": "DynamoDB", "elasticsearch": "Elasticsearch",
            "cassandra": "Cassandra", "mariadb": "MariaDB",
            "snowflake": "Snowflake", "redis": "Redis", "bigquery": "BigQuery",
        }

        devops = {
            "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
            "terraform": "Terraform", "jenkins": "Jenkins", "circleci": "CircleCI",
            "github actions": "GitHub Actions", "ansible": "Ansible", "chef": "Chef",
            "puppet": "Puppet", "cloudformation": "CloudFormation", "pulumi": "Pulumi",
        }

        data_engineering = {
            "kafka": "Kafka", "spark": "Spark", "apache spark": "Spark",
            "hadoop": "Hadoop", "airflow": "Airflow", "dbt": "dbt",
            "dagster": "Dagster", "flink": "Flink",
        }

        ml_ai = {
            "ai": "AI", "ml": "Machine Learning", "machine learning": "Machine Learning",
            "llm": "LLM", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
            "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn",
            "hugging face": "Hugging Face", "transformers": "Transformers",
            "keras": "Keras", "pandas": "Pandas", "numpy": "NumPy",
        }

        # Merge all categories
        self.norm_map = {
            **languages, **web_tech, **frontend, **backend,
            **cloud, **databases, **devops, **data_engineering, **ml_ai,
        }

    def _init_skill_categories(self):
        """Initialize skill categories for analytics."""
        self.skill_categories = {
            "Languages": {"Python", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#", "Objective-C", "R", "Swift", "Java", "PHP", "Ruby"},
            "Web Technology": {"HTML", "CSS"},
            "Frontend": {"React", "Vue", "Angular", "NextJS", "Svelte"},
            "Backend": {"Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot", "Spring", ".NET", "Ruby on Rails", "Laravel"},
            "Cloud": {"AWS", "GCP", "Azure"},
            "Databases": {"SQL", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "DynamoDB", "Elasticsearch", "Cassandra", "MariaDB", "Snowflake", "Redis", "BigQuery"},
            "DevOps": {"Docker", "Kubernetes", "Terraform", "Jenkins", "CircleCI", "GitHub Actions", "Ansible", "Chef", "Puppet", "CloudFormation", "Pulumi"},
            "Data Engineering": {"Kafka", "Spark", "Hadoop", "Airflow", "dbt", "Dagster", "Flink"},
            "ML & AI": {"AI", "Machine Learning", "LLM", "PyTorch", "TensorFlow", "Scikit-learn", "Hugging Face", "Transformers", "Keras", "Pandas", "NumPy"},
        }

    def is_blacklisted(self, skill):
        """Check if a skill is in the blacklist."""
        return skill.lower() in self.blacklist

    def clean_and_categorize(self, skill):
        """
        Clean and normalize a single skill.
        
        Returns:
            Normalized skill name or None if blacklisted.
        """
        if not skill or not isinstance(skill, str):
            return None
    
        # Remove ESCO jargon (parenthetical notes)
        skill = re.sub(r'\s*\(.*\)', '', skill).strip()
        s_lower = skill.lower()

        # Check blacklist
        if self.is_blacklisted(s_lower):
            return None

        # Standardize via norm_map
        if s_lower in self.norm_map:
            return self.norm_map[s_lower]

        # Default: Title case for unknown NER products
        return skill.title()
    
    def get_skill_category(self, skill):
        """Get the category for a given skill."""
        for category, skills in self.skill_categories.items():
            if skill in skills:
                return category
        return "Other"
    
    def classify_role(self, skills, text):
        """
        Classify job role based on skills and text patterns.
        
        Args:
            skills: Pipe-separated skills string
            text: Job posting text
            
        Returns:
            Role classification string
        """
        if not skills or skills == "Unclassified":
            return "Unclassified"
        
        text = str(text).lower()
        skills_lower = str(skills).lower()
        
        # Helper to check if any skill is present
        def has_skills(*skill_list):
            return any(s in skills_lower for s in skill_list)
        
        def has_text(*text_list):
            return any(t in text for t in text_list)
        
        # Frontend Engineer
        if has_skills("react", "vue", "angular", "nextjs", "svelte", "html", "css"):
            if has_text("frontend", "ui", "ux", "web design", "component"):
                return "Frontend Engineer"
            return "Frontend Engineer"
        
        # Backend Engineer
        if has_skills("node.js", "python", "java", "django", "flask", "fastapi", "spring", "laravel", "ruby"):
            if has_skills("react", "vue", "angular"):  # Has both frontend + backend
                return "Full Stack Engineer"
            return "Backend Engineer"
        
        # Full Stack Engineer
        if (has_skills("react", "vue", "angular", "nextjs") and 
            has_skills("node.js", "python", "java", "django", "flask")):
            return "Full Stack Engineer"
        
        # DevOps / SRE Engineer
        if has_skills("kubernetes", "docker", "terraform", "jenkins", "circleci", "ansible", "cloudformation"):
            return "DevOps Engineer"
        
        # Cloud Engineer
        if has_skills("aws", "azure", "gcp", "cloudformation", "terraform"):
            if has_text("architect", "cloud engineer", "infrastructure"):
                return "Cloud Engineer"
            return "Cloud Engineer"
        
        # Data Engineer
        if has_skills("spark", "kafka", "hadoop", "airflow", "dbt", "dagster", "flink"):
            return "Data Engineer"
        
        # ML / AI Engineer
        if has_skills("pytorch", "tensorflow", "scikit-learn", "hugging face", "transformers", "machine learning", "llm"):
            return "ML / AI Engineer"
        
        # Data Scientist
        if has_skills("python", "r", "pandas", "numpy", "scikit-learn"):
            if has_text("data scientist", "analytics", "statistical", "modeling"):
                return "Data Scientist"
        
        # Database Engineer
        if has_skills("postgresql", "mysql", "mongodb", "snowflake", "bigquery", "elasticsearch"):
            return "Database Engineer"
        
        # QA / Test Engineer
        if has_text("qa", "quality assurance", "test automation", "testing"):
            return "QA / Test Engineer"
        
        # Solutions Architect
        if has_text("architect", "solutions architect", "system design", "enterprise"):
            return "Solutions Architect"
        
        # Security Engineer
        if has_text("security", "infosec", "cybersecurity", "penetration"):
            return "Security Engineer"
        
        # Default
        return "General Software Engineer"

    def process_files(self):
        """Process all NLP-extracted CSV files and generate ML-ready output."""
        files = list(self.proc_path.glob("year=*/month=*/NLP_extracted.csv"))
        
        if not files:
            print("❌ No NLP_extracted.csv files found!")
            return

        all_skills = []
        file_data = []
        
        # First pass: collect all skills
        for f in files:
            df = pd.read_csv(f)
            raw_skills = df['skills'].str.split('|').explode().dropna()
            cleaned = [self.clean_and_categorize(s) for s in raw_skills]
            all_skills.extend([c for c in cleaned if c])
            file_data.append((f, df))

        # Determine valid skills (appear 2+ times or in norm_map)
        skill_counts = Counter(all_skills)
        valid_skills = {s for s, count in skill_counts.items() 
                       if count > 1 or s in self.norm_map.values()}
        
        # Second pass: process and save files
        for file_path, df in file_data:
            initial_count = len(df)
            
            # Clean skills column
            df['skills'] = df['skills'].apply(self._process_skills_row, args=(valid_skills,))
            df['has_skills'] = (df['skills'] != "Unclassified").astype(int)
            df['role'] = df.apply(lambda x: self.classify_role(x['skills'], x.get('full_text', '')), axis=1)
            df['skill_category'] = df['skills'].apply(
                lambda x: self.get_skill_category(x.split('|')[0]) if x != "Unclassified" else "Other"
            )
            
            # Save outputs
            output_file = file_path.parent / "ml_ready.csv"
            df.to_csv(output_file, index=False)
            
            # Generate quality report
            self._save_quality_report(file_path.parent, initial_count, df)
            print(f"✅ Processed: {file_path.name} -> {output_file}")

    def _process_skills_row(self, skills_str, valid_skills):
        """Process a single skills string."""
        if pd.isna(skills_str) or skills_str == "Unclassified":
            return "Unclassified"
        
        parts = [self.clean_and_categorize(s) for s in str(skills_str).split('|')]
        filtered = [p for p in parts if p in valid_skills]
        return "|".join(sorted(list(set(filtered)))) if filtered else "Unclassified"

    def _save_quality_report(self, output_dir, initial_count, df):
        """Generate and save data quality report."""
        clean_rows = len(df[df['skills'] != "Unclassified"])
        report = {
            "input_rows": initial_count,
            "output_rows": len(df),
            "clean_rows": clean_rows,
            "unclassified_rows": len(df[df['skills'] == "Unclassified"]),
            "coverage_pct": round((clean_rows / len(df)) * 100, 2),
            "top_skills": Counter(df[df['skills'] != "Unclassified"]['skills'].str.split('|').explode()).most_common(10),
            "top_roles": df['role'].value_counts().head(10).to_dict(),
        }
        
        with open(output_dir / "dq_report.json", 'w') as f:
            json.dump(report, f, indent=4)

if __name__ == "__main__":
    normalizer = Normalizer()
    normalizer.process_files()