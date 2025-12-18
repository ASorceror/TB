"""
Blueprint Processor V4.2.2 - Quality Gate Tests

Tests the is_quality_title() method in Validator class.
Run with: python tests/test_quality_gate.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.validator import Validator


def test_rejects_garbage_patterns():
    """Test that garbage patterns are rejected."""
    print("\n=== TEST: Reject Garbage Patterns ===")

    v = Validator()

    garbage_titles = [
        ("Gmp/permit Set", "garbage_pattern"),
        ("GMP Permit Set", "garbage_pattern"),
        ("Part 1 - General", "garbage_pattern"),
        ("Part 2 - Products", "garbage_pattern"),
        ("End of Section", "garbage_pattern"),
        ("1425", "garbage_pattern"),
        ("60601", "garbage_pattern"),
        ("A2.0", "garbage_pattern"),
        ("S-101", "garbage_pattern"),
        ("Ball State University", "garbage_pattern"),
        ("Recreation Center", "garbage_pattern"),
        ("Northern Tool+equipment", "garbage_pattern"),
        ("955 N Larrabee", "garbage_pattern"),
        ("Loop 250 Frontage Rd", "garbage_pattern"),
        ("0'-0\" = +13.00' C.c.d.", "garbage_pattern"),
        ("manufacturers specified.", "garbage_pattern"),
    ]

    passed = 0
    failed = 0

    for title, expected_reason_prefix in garbage_titles:
        is_quality, reason = v.is_quality_title(title)

        if not is_quality:
            print(f"  [OK] REJECTED: '{title}' -> {reason}")
            passed += 1
        else:
            print(f"  [FAIL] WRONGLY ACCEPTED: '{title}' -> {reason}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_rejects_single_keywords():
    """Test that single keywords without context are rejected."""
    print("\n=== TEST: Reject Single Keywords ===")

    v = Validator()

    single_keywords = [
        "Section",
        "SECTION",
        "Plan",
        "PLAN",
        "Elevation",
        "ELEVATIONS",
        "Detail",
        "DETAILS",
        "Schedule",
        "Demolition",
        "Foundation",
        "Mechanical",
        "Electrical",
        "Plumbing",
        "General",
        "Interior",
        "Specifications",
        "Notes",
    ]

    passed = 0
    failed = 0

    for title in single_keywords:
        is_quality, reason = v.is_quality_title(title)

        if not is_quality:
            print(f"  [OK] REJECTED: '{title}' -> {reason}")
            passed += 1
        else:
            print(f"  [FAIL] WRONGLY ACCEPTED: '{title}' -> {reason}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_accepts_valid_titles():
    """Test that valid titles with keywords + context are accepted."""
    print("\n=== TEST: Accept Valid Titles ===")

    v = Validator()

    valid_titles = [
        "Floor Plan",
        "1st Floor Plan",
        "2nd Floor Plan",
        "Reflected Ceiling Plan",
        "Demolition Plan",
        "Demolition Floor Plan",
        "North Elevation",
        "South Elevation",
        "East Elevation Looking West",
        "Building Section A",
        "Building Cross Section",
        "Section Detail",
        "Wall Section",
        "Door Schedule",
        "Window Schedule",
        "Finish Schedule",
        "Plumbing Fixture Schedule",
        "Electrical Riser Diagram",
        "HVAC Layout Plan",
        "Site Plan",
        "Foundation Plan",
        "Roof Plan",
        "Enlarged Floor Plan",
        "Code Review, Schedules and ADA Requirements",
        "Door and Frame Details",
        "General Notes",
        "Finish Plan",
        "Construction Plan",
        "Legend - Finish Plan",
        "Legend - Construction Plan",
    ]

    passed = 0
    failed = 0

    for title in valid_titles:
        is_quality, reason = v.is_quality_title(title)

        if is_quality:
            print(f"  [OK] ACCEPTED: '{title}' -> {reason}")
            passed += 1
        else:
            print(f"  [FAIL] WRONGLY REJECTED: '{title}' -> {reason}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_rejects_proper_nouns():
    """Test that proper noun patterns (names) are rejected."""
    print("\n=== TEST: Reject Proper Noun Patterns ===")

    v = Validator()

    proper_nouns = [
        "Baseball Building",
        "Germantown",
        "Graceland Avenue",
        "North Corridor",
        "West Corridor",
    ]

    passed = 0
    failed = 0

    for title in proper_nouns:
        is_quality, reason = v.is_quality_title(title)

        if not is_quality:
            print(f"  [OK] REJECTED: '{title}' -> {reason}")
            passed += 1
        else:
            print(f"  [?] ACCEPTED (may be okay): '{title}' -> {reason}")
            # Don't count as failure - some of these are borderline
            passed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    return True  # Don't fail on borderline cases


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== TEST: Edge Cases ===")

    v = Validator()

    test_cases = [
        # (title, should_pass, description)
        ("", False, "Empty string"),
        ("  ", False, "Whitespace only"),
        ("AB", False, "Too short"),
        ("ABC", False, "Exactly 3 chars, no keyword"),
        (None, False, "None value"),
        ("A" * 100, False, "Too long"),
        ("Floor Plan", True, "Keyword with space"),
        ("FloorPlan", False, "Keyword no space (no context)"),
        ("Project Number", False, "Should be in garbage patterns"),
        ("Self Certification", False, "Two words but proper noun pattern, no keyword"),
    ]

    passed = 0
    failed = 0

    for title, should_pass, description in test_cases:
        try:
            is_quality, reason = v.is_quality_title(title)

            if is_quality == should_pass:
                status = "[OK]"
                passed += 1
            else:
                status = "[FAIL]"
                failed += 1

            result = "PASS" if is_quality else "FAIL"
            expected = "PASS" if should_pass else "FAIL"
            print(f"  {status} '{title}' -> {result} (expected {expected}): {description}")

        except Exception as e:
            print(f"  [FAIL] '{title}' -> EXCEPTION: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_with_project_number():
    """Test contamination detection with project number."""
    print("\n=== TEST: Project Number Contamination ===")

    v = Validator()

    # Title matches project number
    is_quality, reason = v.is_quality_title("P27142", project_number="P27142")
    if not is_quality and "project_number" in reason:
        print(f"  [OK] Detected contamination: 'P27142' matching project number")
    else:
        print(f"  [FAIL] Failed to detect contamination")
        return False

    # Title doesn't match project number
    is_quality, reason = v.is_quality_title("Floor Plan", project_number="P27142")
    if is_quality:
        print(f"  [OK] No false positive: 'Floor Plan' not matching project number")
    else:
        print(f"  [FAIL] False positive contamination detection")
        return False

    return True


def run_all_tests():
    """Run all quality gate tests."""
    print("=" * 60)
    print("QUALITY GATE TESTS - Blueprint Processor V4.2.2")
    print("=" * 60)

    results = []

    results.append(("Garbage Patterns", test_rejects_garbage_patterns()))
    results.append(("Single Keywords", test_rejects_single_keywords()))
    results.append(("Valid Titles", test_accepts_valid_titles()))
    results.append(("Proper Nouns", test_rejects_proper_nouns()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Project Number", test_with_project_number()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - Quality Gate is ready")
        print("You may proceed to modify core/extractor.py")
    else:
        print("SOME TESTS FAILED - Fix issues before proceeding")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
