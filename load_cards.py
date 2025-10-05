import json
from typing import Dict, List, Any


def load_modern_atomic(file_path: str = "ModernAtomic.json") -> Dict[str, List[Dict[str, Any]]]:
    """Load the ModernAtomic.json file and return the card data."""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Metadata: {data.get('meta', {})}")
    print(f"Total unique card names: {len(data['data'])}")

    return data['data']


def get_card_count(cards_data: Dict[str, List[Dict[str, Any]]]) -> int:
    """Get total number of card variations (some cards have multiple printings)."""
    total = sum(len(variations) for variations in cards_data.values())
    return total


def clean_card_for_embedding(card: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and simplify a card for vector embedding.
    Removes unnecessary fields and keeps only relevant data.
    """
    # Essential fields for embeddings
    cleaned = {
        'name': card['name'],
        'type': card['type'],
        'types': card.get('types', []),
        'subtypes': card.get('subtypes', []),
        'supertypes': card.get('supertypes', []),
        'text': card.get('text', ''),
        'manaValue': card.get('manaValue', 0),
        'manaCost': card.get('manaCost', ''),
        'colors': card.get('colors', []),
        'colorIdentity': card.get('colorIdentity', []),
        'keywords': card.get('keywords', []),
        'power': card.get('power'),
        'toughness': card.get('toughness'),
        'loyalty': card.get('loyalty'),
        'layout': card.get('layout', ''),
    }

    # Remove None values
    cleaned = {k: v for k, v in cleaned.items() if v is not None}

    return cleaned


def get_unique_cards(cards_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Get one version of each unique card (takes first printing).
    Returns cleaned cards ready for embedding.
    """
    unique_cards = []

    for card_name, variations in cards_data.items():
        if variations:  # Take first variation
            cleaned = clean_card_for_embedding(variations[0])
            unique_cards.append(cleaned)

    return unique_cards


def create_embedding_text(card: Dict[str, Any]) -> str:
    """
    Create a text representation of a card for embedding.
    Combines relevant fields into a single string.
    """
    parts = []

    # Name
    parts.append(f"Name: {card['name']}")

    # Type line
    parts.append(f"Type: {card['type']}")

    # Mana cost
    if card.get('manaCost'):
        parts.append(f"Mana Cost: {card['manaCost']}")

    # Card text
    if card.get('text'):
        parts.append(f"Text: {card['text']}")

    # Power/Toughness for creatures
    if card.get('power') and card.get('toughness'):
        parts.append(f"P/T: {card['power']}/{card['toughness']}")

    # Loyalty for planeswalkers
    if card.get('loyalty'):
        parts.append(f"Loyalty: {card['loyalty']}")

    # Keywords
    if card.get('keywords'):
        parts.append(f"Keywords: {', '.join(card['keywords'])}")

    return "\n".join(parts)


if __name__ == "__main__":
    # Load the data
    cards_data = load_modern_atomic()

    print(f"Total card variations: {get_card_count(cards_data)}")

    # Get unique cards
    unique_cards = get_unique_cards(cards_data)
    print(f"Unique cards after cleaning: {len(unique_cards)}")

    # Example: Show first card
    if unique_cards:
        print("\n" + "="*80)
        print("Example card (cleaned):")
        print(json.dumps(unique_cards[0], indent=2))

        print("\n" + "="*80)
        print("Example embedding text:")
        print(create_embedding_text(unique_cards[0]))

    # Save cleaned data
    print("\nSaving cleaned cards to 'cleaned_cards.json'...")
    with open('cleaned_cards.json', 'w', encoding='utf-8') as f:
        json.dump(unique_cards, f, indent=2)

    print("Done!")
