"""
Title Block Detection via Transition Detection

Key insight from the column profile:
- Drawing area: nearly ZERO consistent edges
- Title block: ABOVE ZERO consistent edges (even sparse areas)

The LEFT BORDER of the title block is where consistent edges FIRST APPEAR
(transition from zero to non-zero).

Approach:
1. Get common edges
2. Calculate column edge counts
3. Find where edge count rises above a LOW threshold (based on drawing baseline)
4. That transition point = title block left border
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler


def find_title_block_transition(pdf_path: Path, sample_pages: list = None, dpi: int = 100):
    """
    Find title block by detecting transition from zero to non-zero edges.
    """
    output_dir = Path(r"C:\tb\blueprint_processor\output\research")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = pdf_path.stem[:30].replace(' ', '_')

    with PDFHandler(pdf_path) as handler:
        total_pages = handler.page_count

        if sample_pages is None:
            start = 2 if total_pages > 5 else 1
            if total_pages > 20:
                sample_pages = [2, 5, 8, 12, 15]
            else:
                sample_pages = list(range(start, min(start + 5, total_pages + 1)))

        print(f"\n{'='*60}")
        print(f"PDF: {pdf_path.name}")
        print(f"Total pages: {total_pages}")
        print(f"Sample pages: {sample_pages}")

        # Step 1: Render and extract edges
        edge_images = []
        target_size = None

        for page_num in sample_pages:
            img = handler.get_page_image(page_num - 1, dpi=dpi)

            if target_size is None:
                target_size = img.size
            elif img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            gray = img.convert('L')
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)
            binary = (edges_array > 25).astype(np.uint8) * 255
            edge_images.append(binary)

        print(f"  Processed {len(edge_images)} pages")

        # Step 2: Find common edges
        stack = np.stack(edge_images, axis=0)
        common_edges = np.all(stack > 128, axis=0).astype(np.uint8) * 255

        height, width = common_edges.shape

        # Step 3: Calculate column edge counts
        col_edge_count = np.sum(common_edges > 128, axis=0)

        # Smooth to reduce noise
        from scipy.ndimage import uniform_filter1d
        col_smoothed = uniform_filter1d(col_edge_count.astype(float), size=10)

        # Step 4: Analyze the drawing area baseline
        # Use the LEFT portion (0-50%) to establish baseline
        left_portion = col_smoothed[:int(width * 0.5)]

        # The baseline is the typical edge count in the drawing area
        # Use median to be robust to outliers (like the page border)
        baseline = np.median(left_portion)

        # Use percentile to find a reasonable threshold
        # 95th percentile of left portion = noise level in drawing area
        noise_level = np.percentile(left_portion, 95)

        print(f"\n  Baseline Analysis (left 50% of page):")
        print(f"    Median: {baseline:.1f}")
        print(f"    95th percentile (noise level): {noise_level:.1f}")

        # Step 5: Search in right portion for transition
        # Transition = where edge count rises ABOVE noise level
        search_start = int(width * 0.60)  # Start at 60%
        search_end = int(width * 0.97)    # End at 97%

        # Threshold: significantly above noise level
        threshold = max(noise_level * 2, 20)  # At least 2x noise, minimum 20

        print(f"  Transition threshold: {threshold:.1f}")

        # Find the first column where edge count exceeds threshold
        # and STAYS above threshold (not just a spike)
        window_size = 20  # Must be above threshold for 20 consecutive columns

        transition_col = None
        for i in range(search_start, search_end - window_size):
            window = col_smoothed[i:i + window_size]
            if np.all(window > threshold):
                transition_col = i
                break

        if transition_col is not None:
            x1 = transition_col

            # Find where density is highest (right edge of title block structure)
            right_portion = col_smoothed[x1:search_end]
            x2 = x1 + len(right_portion) - 1

            # Alternatively, find where density drops back to baseline
            # (if there's whitespace after title block)
            for i in range(len(right_portion) - window_size, 0, -1):
                if col_smoothed[x1 + i] > threshold:
                    x2 = x1 + i
                    break

            # Calculate percentages
            x1_pct = x1 / width
            x2_pct = x2 / width
            width_pct = (x2 - x1) / width

            print(f"\n  DETECTED TITLE BLOCK:")
            print(f"    Transition at column: {transition_col}")
            print(f"    Edge count at transition: {col_smoothed[transition_col]:.1f}")
            print(f"    Left edge (x1): {x1_pct:.3f}")
            print(f"    Right edge (x2): {x2_pct:.3f}")
            print(f"    Width: {width_pct*100:.1f}% of page")

            # Save visualization
            first_page = handler.get_page_image(sample_pages[0] - 1, dpi=dpi)
            if first_page.size != target_size:
                first_page = first_page.resize(target_size, Image.Resampling.LANCZOS)

            result_img = first_page.convert('RGB')
            draw = ImageDraw.Draw(result_img)

            # Green line at left boundary (transition point)
            draw.line([(x1, 0), (x1, height)], fill='green', width=3)

            # Red box for full title block
            draw.rectangle([x1, 0, x2, height], outline='red', width=2)

            detection_path = output_dir / f"{pdf_name}_transition_detected.png"
            result_img.save(detection_path)
            print(f"  Saved: {detection_path}")

            # Save transition profile
            save_transition_profile(col_smoothed, search_start, search_end, x1, threshold, baseline,
                                  output_dir / f"{pdf_name}_transition_profile.png")

            return {
                'bbox_percent': {
                    'x1': x1_pct,
                    'y1': 0.0,
                    'x2': x2_pct,
                    'y2': 1.0
                },
                'width_percent': width_pct,
            }
        else:
            print("  WARNING: Could not detect transition")
            save_transition_profile(col_smoothed, search_start, search_end, None, threshold, baseline,
                                  output_dir / f"{pdf_name}_transition_profile.png")
            return None


def save_transition_profile(col_counts, search_start, search_end, detected_x, threshold, baseline, output_path):
    """Save transition profile plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Full page profile
        axes[0].plot(col_counts)
        axes[0].axvline(x=search_start, color='orange', linestyle='--', label='Search start (60%)')
        axes[0].axvline(x=search_end, color='orange', linestyle='--', label='Search end (97%)')
        if detected_x:
            axes[0].axvline(x=detected_x, color='green', linestyle='-', linewidth=2,
                          label=f'Transition x1={detected_x}')
        axes[0].axhline(y=baseline, color='gray', linestyle=':', label=f'Baseline={baseline:.0f}')
        axes[0].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold:.0f}')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Edge pixel count (smoothed)')
        axes[0].set_title('Column Edge Profile (full page)')
        axes[0].legend()

        # Zoomed search region
        search_counts = col_counts[search_start:search_end]
        axes[1].plot(range(search_start, search_end), search_counts)
        axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold:.0f}')
        if detected_x:
            axes[1].axvline(x=detected_x, color='green', linestyle='-', linewidth=2,
                          label=f'Transition x1={detected_x}')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Edge pixel count (smoothed)')
        axes[1].set_title('Transition Detection (search region)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Could not save plot: {e}")


def main():
    """Test transition detection."""
    test_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    test_pdfs = [
        "0_full_permit_set_chiro_one_evergreen_park.pdf",
        "3-7-25 Kriser's Highand Final Set.pdf",
        "955 Larrabee - Issue For Pricing Drawings - 05-02-2025.pdf",
        "2018-1203 Ellinwood GMP_Permit Set.pdf",
        "Janesville Nissan Full set Issued for Bids.pdf",
    ]

    print("="*70)
    print("TRANSITION: Title Block Detection via Transition Detection")
    print("="*70)

    results = []

    for pdf_name in test_pdfs:
        pdf_path = test_dir / pdf_name
        if pdf_path.exists():
            try:
                result = find_title_block_transition(pdf_path)
                if result:
                    results.append({
                        'pdf': pdf_name[:40],
                        'x1': result['bbox_percent']['x1'],
                        'width': result['width_percent']
                    })
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - TRANSITION DETECTION")
    print("="*70)
    print(f"{'PDF':<45} {'x1':>8} {'Width':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['pdf']:<45} {r['x1']:>7.3f} {r['width']*100:>7.1f}%")

    print("\nOutput saved to: C:\\tb\\blueprint_processor\\output\\research")


if __name__ == "__main__":
    main()
