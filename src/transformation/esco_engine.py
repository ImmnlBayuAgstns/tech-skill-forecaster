import pandas as pd
import ahocorasick
import re
from pathlib import Path

class ESCOTaxonomyEngine:
    def __init__(self, skills_csv, digital_filter_csv):
        self.skills_csv = Path(skills_csv)
        self.filter_csv = Path(digital_filter_csv)
        self.automaton = ahocorasick.Automaton()
        self.taxonomy_map = {} # Maps synonym -> Preferred Label
        
        self._build_engine()

    def _build_engine(self):
        """Loads ESCO data and builds the Aho-Corasick Automaton."""
        if not self.skills_csv.exists() or not self.filter_csv.exists():
            print("⚠️ ESCO Files missing. Falling back to empty taxonomy.")
            return

        # 1. Load and Filter for Digital Skills only
        all_skills = pd.read_csv(self.skills_csv)
        digital_uris = pd.read_csv(self.filter_csv)['conceptUri']
        tech_df = all_skills[all_skills['conceptUri'].isin(digital_uris)]

        # 2. Build the Mapping (Synonyms to Preferred Labels)
        for _, row in tech_df.iterrows():
            pref = row['preferredLabel']
            # Map the main name
            self._add_to_automaton(pref.lower(), pref)
            
            # Map all alternative labels (synonyms)
            if pd.notna(row['altLabels']):
                for alt in row['altLabels'].split('\n'):
                    self._add_to_automaton(alt.lower().strip(), pref)

        # 3. Finalize the search tree
        self.automaton.make_automaton()
        print(f"✅ ESCO Engine Ready: {len(tech_df)} tech skills loaded.")

    def _add_to_automaton(self, key, value):
        if key not in self.automaton:
            self.automaton.add_word(key, value)

    def extract(self, text):
        """Single-pass extraction with word-boundary protection."""
        if not text or not isinstance(text, str):
            return []

        text_low = text.lower()
        found = set()

        for end_index, pref_label in self.automaton.iter(text_low):
            # Calculate start index to verify word boundaries
            start_index = end_index - len(pref_label) + 1
            if self._is_exact_match(text_low, start_index, end_index):
                found.add(pref_label)
        
        return list(found)

    def _is_exact_match(self, text, start, end):
        """Prevents 'Java' matching inside 'JavaScript'."""
        pattern = r'[\w]'
        if start > 0 and re.match(pattern, text[start-1]):
            return False
        if end < len(text) - 1 and re.match(pattern, text[end+1]):
            return False
        return True