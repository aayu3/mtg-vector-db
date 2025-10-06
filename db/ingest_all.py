"""
Master ingestion script that runs all ingestion processes sequentially.
Ingests glossary, rules, and cards into the MTG vector database.
"""

import sys
import time
from ingest_glossary import ingest_glossary
from ingest_rules import ingest_rules
from ingest_cards import ingest_cards


def main():
    """Run all ingestion scripts in sequence."""
    print("=" * 80)
    print("MTG Vector Database - Full Ingestion")
    print("=" * 80)
    print("\nThis script will ingest all MTG data into the database:")
    print("  1. Glossary (~700 entries)")
    print("  2. Rules (~2000+ rules)")
    print("  3. Cards (~30000+ cards)")
    print("\nEstimated time: 45-75 minutes")
    print("=" * 80)

    # Configuration (shared across all scripts)
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "mtg_vectors",
        "user": "postgres",
        "password": "postgres"
    }

    ollama_config = {
        "base_url": "http://localhost:11434",
        "model": "embeddinggemma:latest"
    }

    start_time = time.time()
    results = {
        "glossary": False,
        "rules": False,
        "cards": False
    }

    # Step 1: Ingest Glossary
    print("\n" + "=" * 80)
    print("STEP 1/3: GLOSSARY INGESTION")
    print("=" * 80)
    try:
        glossary_start = time.time()
        results["glossary"] = ingest_glossary(
            glossary_file="../rulesCleaning/MagicRulesGlossary.txt",
            db_config=db_config,
            ollama_config=ollama_config,
            batch_size=50
        )
        glossary_time = time.time() - glossary_start
        print(f"\n✓ Glossary ingestion completed in {glossary_time/60:.1f} minutes")
    except Exception as e:
        print(f"\n✗ Glossary ingestion failed: {e}")
        results["glossary"] = False

    # Step 2: Ingest Rules
    print("\n" + "=" * 80)
    print("STEP 2/3: RULES INGESTION")
    print("=" * 80)
    try:
        rules_start = time.time()
        results["rules"] = ingest_rules(
            json_file="../rulesCleaning/rules_individual.json",
            db_config=db_config,
            ollama_config=ollama_config,
            batch_size=100
        )
        rules_time = time.time() - rules_start
        print(f"\n✓ Rules ingestion completed in {rules_time/60:.1f} minutes")
    except Exception as e:
        print(f"\n✗ Rules ingestion failed: {e}")
        results["rules"] = False

    # Step 3: Ingest Cards
    print("\n" + "=" * 80)
    print("STEP 3/3: CARDS INGESTION")
    print("=" * 80)
    try:
        cards_start = time.time()
        results["cards"] = ingest_cards(
            json_file="../cardsCleaning/ModernAtomic_cleaned.json",
            db_config=db_config,
            ollama_config=ollama_config,
            batch_size=100
        )
        cards_time = time.time() - cards_start
        print(f"\n✓ Cards ingestion completed in {cards_time/60:.1f} minutes")
    except Exception as e:
        print(f"\n✗ Cards ingestion failed: {e}")
        results["cards"] = False

    # Final Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\nResults:")
    print(f"  Glossary: {'✓ SUCCESS' if results['glossary'] else '✗ FAILED'}")
    print(f"  Rules:    {'✓ SUCCESS' if results['rules'] else '✗ FAILED'}")
    print(f"  Cards:    {'✓ SUCCESS' if results['cards'] else '✗ FAILED'}")

    # Exit with appropriate code
    all_success = all(results.values())
    if all_success:
        print("\n🎉 All ingestion processes completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some ingestion processes failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
