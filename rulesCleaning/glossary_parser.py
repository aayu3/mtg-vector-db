import json
import re
from typing import List, Dict, Any


class GlossaryParser:
    """Parser for MTG Rules Glossary text file."""

    def __init__(self, glossary_file: str):
        self.glossary_file = glossary_file
        self.entries = []

    def parse(self) -> List[Dict[str, Any]]:
        """Parse the glossary file into structured entries."""
        print(f"Parsing {self.glossary_file}...")

        with open(self.glossary_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        # Split by double newlines to get individual entries
        # Each entry starts with a term and follows with definition
        raw_entries = content.strip().split('\n\n')

        for raw_entry in raw_entries:
            if not raw_entry.strip():
                continue

            # Split into lines
            lines = raw_entry.strip().split('\n')

            if len(lines) < 2:
                # Skip malformed entries
                continue

            # First line is the term
            term = lines[0].strip()

            # Rest is the definition (may span multiple lines)
            definition = '\n'.join(lines[1:]).strip()

            # Extract rule references from the definition
            related_rules = self._extract_rule_references(definition)

            entry = {
                'term': term,
                'definition': definition,
                'related_rules': related_rules
            }

            self.entries.append(entry)

        print(f"Parsed {len(self.entries)} glossary entries")
        return self.entries

    def _extract_rule_references(self, text: str) -> List[str]:
        """Extract rule number references from text."""
        # Pattern matches "rule XXX" or "rule XXX.XXX" etc.
        pattern = r'rule\s+(\d+(?:\.\d+[a-z]*)?)'
        matches = re.findall(pattern, text, re.IGNORECASE)

        # Also check for "section X" references
        section_pattern = r'section\s+(\d+)'
        section_matches = re.findall(section_pattern, text, re.IGNORECASE)

        # Combine and deduplicate
        all_refs = list(set(matches + section_matches))
        return sorted(all_refs)

    def save(self, output_file: str) -> None:
        """Save parsed glossary to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(self.entries)} glossary entries to {output_file}")


def main():
    """Main execution."""
    parser = GlossaryParser('../rulesCleaning/MagicRulesGlossary.txt')
    entries = parser.parse()
    parser.save('glossary.json')

    # Show sample entries
    print("\n" + "=" * 80)
    print("Sample Glossary Entries:")
    print("=" * 80)

    for entry in entries[:3]:
        print(f"\nTerm: {entry['term']}")
        print(f"Definition: {entry['definition'][:200]}...")
        if entry['related_rules']:
            print(f"Related Rules: {', '.join(entry['related_rules'])}")


if __name__ == "__main__":
    main()
