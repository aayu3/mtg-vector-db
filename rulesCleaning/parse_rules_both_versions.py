import json
import re
from typing import List, Dict, Any, Set
from collections import defaultdict


class RulesParser:
    """Parser that creates both individual and combined rule versions."""

    def __init__(self, index_file: str, numbered_file: str):
        self.index_file = index_file
        self.numbered_file = numbered_file
        self.index_lines = set()
        self.all_rules = []  # Every single rule line

    def parse(self) -> None:
        """Parse all rules."""
        print("Loading index lines to skip...")
        self._load_index()

        print("Parsing numbered rules...")
        self._parse_all_rules()

    def _load_index(self) -> None:
        """Load all lines from index file to skip them."""
        with open(self.index_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.index_lines.add(line)

        print(f"  Loaded {len(self.index_lines)} index lines to skip")

    def _parse_all_rules(self) -> None:
        """Parse each rule line from numbered file."""
        with open(self.numbered_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip()

            # Skip empty lines
            if not line:
                continue

            # Skip if this line is in the index (section/main rule header)
            if line in self.index_lines:
                continue

            # Split by first space to get rule number
            # Format: "100.1. Text" (main rule) or "100.1a Text" (subrule)
            parts = line.split(' ', 1)
            if len(parts) == 2:
                rule_number_raw = parts[0]
                rule_text = parts[1]

                # Remove trailing period if present
                if rule_number_raw.endswith('.'):
                    rule_number = rule_number_raw[:-1]
                else:
                    rule_number = rule_number_raw

                # Only include if it looks like a rule number (contains digits and a period)
                if '.' in rule_number and any(c.isdigit() for c in rule_number):
                    rule_entry = {
                        'rule_number': rule_number,
                        'text': rule_text
                    }

                    self.all_rules.append(rule_entry)

        print(f"  Extracted {len(self.all_rules)} total rules")

    def get_section_info(self, rule_number: str) -> Dict[str, str]:
        """Determine section info from rule number."""
        main_num = rule_number.split('.')[0]
        section_num = main_num[0]

        section_names = {
            '1': 'Game Concepts',
            '2': 'Parts of a Card',
            '3': 'Card Types',
            '4': 'Zones',
            '5': 'Turn Structure',
            '6': 'Spells, Abilities, and Effects',
            '7': 'Additional Rules',
            '8': 'Multiplayer Rules',
            '9': 'Casual Variants'
        }

        return {
            'section_number': section_num,
            'section_name': section_names.get(section_num, 'Unknown')
        }

    def is_main_rule(self, rule_number: str) -> bool:
        """Check if this is a main rule (e.g., 100.1) vs subrule (e.g., 100.1a)."""
        return not re.search(r'[a-z]$', rule_number)

    def get_parent_rule(self, rule_number: str) -> str:
        """Get immediate parent rule number.

        100.1a -> 100.1
        100.1 -> None (main rules don't have a parent rule, only a section)
        """
        # If it has a letter suffix, remove it to get parent rule
        if re.search(r'[a-z]$', rule_number):
            return re.sub(r'[a-z]+$', '', rule_number)
        else:
            # Main rule has no parent rule
            return None

    def get_section_parent(self, rule_number: str) -> str:
        """Get section parent number.

        100.1a -> 100
        100.1 -> 100
        """
        return rule_number.split('.')[0]

    def create_individual_rules(self) -> List[Dict[str, Any]]:
        """Create version with each rule as individual entry."""
        individual = []

        for rule in self.all_rules:
            rule_num = rule['rule_number']
            section_info = self.get_section_info(rule_num)
            parent_rule = self.get_parent_rule(rule_num)
            section_parent = self.get_section_parent(rule_num)
            is_main = self.is_main_rule(rule_num)

            entry = {
                'rule_number': rule_num,
                'text': rule['text'],
                'rule_type': 'main_rule' if is_main else 'subrule',
                'section_parent': section_parent,
                'section_number': section_info['section_number'],
                'section_name': section_info['section_name']
            }

            # Only add parent_rule if it exists (subrules have parent rules)
            if parent_rule:
                entry['parent_rule'] = parent_rule

            individual.append(entry)

        return individual

    def create_combined_rules(self) -> List[Dict[str, Any]]:
        """Create version where subrules are nested under main rules."""
        # Group subrules by their parent
        grouped = defaultdict(list)
        main_rules = {}

        for rule in self.all_rules:
            rule_num = rule['rule_number']

            if self.is_main_rule(rule_num):
                # This is a main rule (e.g., 100.1)
                section_info = self.get_section_info(rule_num)
                section_parent = self.get_section_parent(rule_num)

                main_rules[rule_num] = {
                    'rule_number': rule_num,
                    'text': rule['text'],
                    'rule_type': 'main_rule',
                    'section_parent': section_parent,
                    'section_number': section_info['section_number'],
                    'section_name': section_info['section_name'],
                    'subrules': []
                }
            else:
                # This is a subrule (e.g., 100.1a)
                parent_rule = self.get_parent_rule(rule_num)
                section_parent = self.get_section_parent(rule_num)
                section_info = self.get_section_info(rule_num)

                subrule = {
                    'rule_number': rule_num,
                    'text': rule['text'],
                    'rule_type': 'subrule',
                    'section_parent': section_parent,
                    'section_number': section_info['section_number'],
                    'section_name': section_info['section_name']
                }

                grouped[parent_rule].append(subrule)

        # Attach subrules to their parents
        for parent_num, subrules in grouped.items():
            if parent_num in main_rules:
                main_rules[parent_num]['subrules'] = subrules

        # Convert to list and sort
        combined = list(main_rules.values())
        combined.sort(key=lambda x: self._sort_key(x['rule_number']))

        return combined

    def _sort_key(self, rule_num: str):
        """Create sort key for rule numbers."""
        match = re.match(r'^(\d+)\.(\d+)([a-z]*)$', rule_num)
        if match:
            main = int(match.group(1))
            sub = int(match.group(2))
            letter = match.group(3) if match.group(3) else ''
            return (main, sub, letter)
        return (0, 0, '')

    def create_embedding_text_individual(self, rule: Dict[str, Any]) -> str:
        """Create embedding text for individual rule."""
        parts = [
            f"Rule {rule['rule_number']}",
            f"Type: {rule['rule_type']}",
            f"Section: {rule['section_name']}",
            f"Parent: {rule['parent_rule']}",
            f"Text: {rule['text']}"
        ]
        return '\n'.join(parts)

    def create_embedding_text_combined(self, rule: Dict[str, Any]) -> str:
        """Create embedding text for combined rule (includes subrules)."""
        parts = [
            f"Rule {rule['rule_number']}",
            f"Section: {rule['section_name']}",
            f"Text: {rule['text']}"
        ]

        if rule.get('subrules'):
            subrule_texts = []
            for subrule in rule['subrules']:
                subrule_texts.append(f"{subrule['rule_number']}: {subrule['text']}")
            parts.append(f"Subrules:\n" + '\n'.join(subrule_texts))

        return '\n'.join(parts)

    def save_individual(self, output_path: str) -> List[Dict[str, Any]]:
        """Save individual rules version."""
        rules = self.create_individual_rules()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(rules)} individual rules to {output_path}")
        return rules

    def save_combined(self, output_path: str) -> List[Dict[str, Any]]:
        """Save combined rules version."""
        rules = self.create_combined_rules()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)

        total_subrules = sum(len(r.get('subrules', [])) for r in rules)
        print(f"Saved {len(rules)} main rules with {total_subrules} subrules to {output_path}")
        return rules


def main():
    """Main execution."""
    print("=" * 80)
    print("Magic Rules Parser - Individual & Combined Versions")
    print("=" * 80)

    parser = RulesParser('MagicRulesIndex.txt', 'MagicRulesNumbered.txt')
    parser.parse()

    # Create both versions
    print("\n" + "=" * 80)
    print("Creating individual rules version...")
    individual_rules = parser.save_individual('rules_individual.json')

    print("\n" + "=" * 80)
    print("Creating combined rules version...")
    combined_rules = parser.save_combined('rules_combined.json')

    # Show examples
    print("\n" + "=" * 80)
    print("INDIVIDUAL VERSION Examples:")
    print("=" * 80)

    # Show main rule
    main_example = next((r for r in individual_rules if r['rule_type'] == 'main_rule'), None)
    if main_example:
        print("\nMain Rule:")
        print(json.dumps(main_example, indent=2))

    # Show subrule
    sub_example = next((r for r in individual_rules if r['rule_type'] == 'subrule'), None)
    if sub_example:
        print("\nSubrule:")
        print(json.dumps(sub_example, indent=2))

    print("\n" + "=" * 80)
    print("COMBINED VERSION Example:")
    print("=" * 80)

    if combined_rules and combined_rules[0].get('subrules'):
        print(json.dumps(combined_rules[0], indent=2)[:500] + "...")

    # Statistics
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)

    main_count = sum(1 for r in individual_rules if r['rule_type'] == 'main_rule')
    sub_count = sum(1 for r in individual_rules if r['rule_type'] == 'subrule')

    print(f"\nIndividual version:")
    print(f"  Total entries: {len(individual_rules)}")
    print(f"  Main rules: {main_count}")
    print(f"  Subrules: {sub_count}")

    print(f"\nCombined version:")
    print(f"  Main rules: {len(combined_rules)}")
    print(f"  Total subrules nested: {sum(len(r.get('subrules', [])) for r in combined_rules)}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
