import json

def clean_card_data(card):
    """Remove unnecessary fields from a card object."""
    fields_to_remove = [
        'edhrecRank',
        'edhrecSaltiness',
        'firstPrinting',
        'foreignData',
        'purchaseUrls',
        'subsets'
    ]

    for field in fields_to_remove:
        card.pop(field, None)

    return card

def clean_mtg_json(input_file, output_file):
    """Clean MTG JSON file by removing unnecessary data from cards."""
    print(f"Loading {input_file}...") 
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data['data'])} card entries...")

    # Clean each card in the data
    cleaned_count = 0
    for card_array in data['data'].values():
        for card in card_array:
            clean_card_data(card)
            cleaned_count += 1

    print(f"Cleaned {cleaned_count} cards")
    print(f"Removing meta wrapper, keeping only card data...")

    # Extract just the data object, removing the meta wrapper
    cleaned_data = data['data']

    print(f"Writing to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print("Done!")

if __name__ == "__main__":
    input_file = "ModernAtomic.json"
    output_file = "ModernAtomic_cleaned.json"

    clean_mtg_json(input_file, output_file)
