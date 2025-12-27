"""
Cleanup script for ground truth project names.
Normalizes project names per PDF based on most common variant.
"""

import csv
import re
from collections import Counter
from pathlib import Path

# Manual canonical names for each PDF (based on analysis)
CANONICAL_NAMES = {
    "0_full_permit_set_chiro_one_evergreen_park.pdf": "CHIRO ONE WELLNESS CENTER",
    "18222 midland tx - final 2-19-19 rev 1.pdf": "NORTHERN TOOL & EQUIPMENT",
    "2018-1203 Ellinwood GMP_Permit Set.pdf": "ELLINWOOD APARTMENTS",
    "2019.10.23 CHICAGO TABERNACLE KILPATRICK OFFICES.pdf": "CHICAGO TABERNACLE - KILPATRICK OFFICES",
    "2025-05-16 Woodstock Recreation Center IFB-IFP_Drawings.pdf": "WOODSTOCK RECREATION CENTER",
    "2025_0319_CM RFP_drawings.pdf": "TRUCK COUNTRY GERMANTOWN",
    "3-7-25 Kriser's Highand Final Set.pdf": "KRISER'S NATURAL PET",
    "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf": "955 N LARRABEE",
    "B5155_FullSet.pdf": "THE PEARL",
    "Baseball And Softball Locker Room Buildings_5970807_12032025.pdf": "BALL STATE UNIVERSITY - BASEBALL & SOFTBALL LOCKER ROOM BUILDINGS",
    "Cascade East_2-17-20 permits.pdf": "CASCADE EAST",
    "DQ Matteson A Permit Drawings.pdf": "DAIRY QUEEN LIBERTY PLAZA",
    "Full Set_080219.pdf": "PLANET FITNESS",
    "Janesville Nissan Full set Issued for Bids.pdf": "JANESVILLE NISSAN",
    "Senju Office TI Arch Permit Set 2.4.20.pdf": "SENJU AMERICA INC",
    "marshalls_bid_set_11-26-19.pdf": "MARSHALLS - SHOPS ON MAIN",
    "quarry sally beauty plans 2020 05 22.pdf": "SALLY BEAUTY",
}

# Patterns that indicate a value is NOT a project name
NON_PROJECT_PATTERNS = [
    r'^B\d+-\d+\s+Build-out',  # Scope descriptions like "B1-1 Build-out..."
    r'^\d+\s+sf\s+',  # Square footage descriptions
    r'^New Tenant Improvement$',  # Generic description
    r'^\d{4,}\s+\w+\s+(Road|Street|Ave|Blvd)',  # Addresses
    r'^(N|S|E|W)\d+\w*\d+\s+',  # Address codes like "N128W21795..."
]


def is_valid_project_name(name: str) -> bool:
    """Check if a string looks like a valid project name."""
    if not name or len(name) < 3:
        return False
    if len(name) > 100:  # Too long, probably a description
        return False
    for pattern in NON_PROJECT_PATTERNS:
        if re.match(pattern, name, re.IGNORECASE):
            return False
    return True


def normalize_project_name(name: str) -> str:
    """Normalize a project name for comparison."""
    if not name:
        return ""
    # Uppercase
    name = name.upper()
    # Normalize symbols
    name = name.replace("+", "&").replace("  ", " ")
    # Remove trailing punctuation
    name = name.strip().rstrip(".,;:")
    return name


def get_canonical_name(pdf_filename: str, project_names: list) -> str:
    """Determine the canonical project name for a PDF."""
    # First check manual overrides
    if pdf_filename in CANONICAL_NAMES:
        return CANONICAL_NAMES[pdf_filename]

    # Filter valid names
    valid_names = [n for n in project_names if is_valid_project_name(n)]
    if not valid_names:
        return ""

    # Count normalized versions
    normalized_counts = Counter()
    for name in valid_names:
        norm = normalize_project_name(name)
        if norm:
            normalized_counts[norm] += 1

    if not normalized_counts:
        return ""

    # Return the most common (keeping original casing from first occurrence)
    most_common_norm = normalized_counts.most_common(1)[0][0]

    # Find the best original version (prefer uppercase)
    for name in valid_names:
        if normalize_project_name(name) == most_common_norm:
            if name.isupper():
                return name

    # Fall back to first match
    for name in valid_names:
        if normalize_project_name(name) == most_common_norm:
            return name.upper()

    return ""


def cleanup_ground_truth(input_path: Path, output_path: Path):
    """Clean up project names in ground truth CSV."""

    # Load all rows
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Group project names by PDF
    pdf_projects = {}
    for row in rows:
        pdf = row['pdf_filename']
        proj = row.get('project_name', '') or ''
        if pdf not in pdf_projects:
            pdf_projects[pdf] = []
        if proj:
            pdf_projects[pdf].append(proj)

    # Determine canonical name for each PDF
    canonical_map = {}
    print("=" * 60)
    print("CANONICAL PROJECT NAMES")
    print("=" * 60)
    for pdf in sorted(pdf_projects.keys()):
        canonical = get_canonical_name(pdf, pdf_projects[pdf])
        canonical_map[pdf] = canonical
        print(f"{pdf[:45]:<45} -> {canonical}")

    # Update rows
    changes = 0
    for row in rows:
        pdf = row['pdf_filename']
        old_name = row.get('project_name', '') or ''
        new_name = canonical_map.get(pdf, '')

        if old_name != new_name:
            row['project_name'] = new_name
            changes += 1

    # Write output
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(rows)}")
    print(f"Rows updated: {changes}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    input_file = Path("output/ground_truth_v2.csv")
    output_file = Path("output/ground_truth_v2_cleaned.csv")

    cleanup_ground_truth(input_file, output_file)
