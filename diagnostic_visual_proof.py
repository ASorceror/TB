"""
Visual Proof Diagnostic - Shows EXACTLY what crops are used at each step.

Saves images for:
1. Full page (raw)
2. Full page with title block bbox overlay
3. Title block crop
4. Sheet identification zone crop (from title block)
5. Each zone with labeled overlay

This provides complete visual proof of what the AI Vision detected
and what crops are being used.
"""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))

from core.pdf_handler import PDFHandler
from core.title_block_discovery import TitleBlockDiscovery, calculate_sample_pages


def draw_labeled_bbox(image: Image.Image, bbox: tuple, label: str, color: str = 'red') -> Image.Image:
    """Draw a labeled bounding box on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Draw label background
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x1, y1 - 25), label, fill='white', font=font)

    return img


def run_visual_proof():
    """Run comprehensive visual proof diagnostic."""

    output_dir = Path(r"C:\tb\blueprint_processor\output\visual_proof")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("VISUAL PROOF DIAGNOSTIC")
    print(f"Output: {run_dir}")
    print(f"{'='*70}\n")

    test_data_dir = Path(r"C:\Hybrid-Extraction-Test\Test Blueprints")

    # Get all PDFs in the test folder
    all_pdfs = sorted(test_data_dir.glob("*.pdf"))
    print(f"Found {len(all_pdfs)} PDFs to process\n")

    # Initialize discovery
    telemetry_dir = run_dir / 'telemetry'
    telemetry_dir.mkdir(exist_ok=True)
    discovery = TitleBlockDiscovery(cache_dir=telemetry_dir)

    # Summary tracking
    summary = []

    for pdf_idx, pdf_path in enumerate(all_pdfs, 1):
        pdf_name = pdf_path.name

        # Create output folder
        pdf_safe = pdf_name.replace(' ', '_').replace('.pdf', '')[:40]
        pdf_dir = run_dir / pdf_safe
        pdf_dir.mkdir(exist_ok=True)

        print(f"\n[{pdf_idx}/{len(all_pdfs)}] PDF: {pdf_name}")
        print(f"Output: {pdf_dir}")

        # Compute hash
        with open(pdf_path, 'rb') as f:
            pdf_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        try:
            with PDFHandler(pdf_path) as handler:
                total_pages = handler.page_count

                # Sample pages: first 3 pages, or all if fewer
                pages = list(range(1, min(4, total_pages + 1)))
                print(f"  Total pages: {total_pages}, sampling: {pages}")

                # Step 1: Run discovery
                print(f"\n  Step 1: Running discovery...")
                discovery.discover(handler, pdf_hash, force_refresh=True)
                telemetry = discovery.get_current_telemetry()

                # Save telemetry as JSON
                telemetry_path = pdf_dir / "telemetry.json"
                with open(telemetry_path, 'w') as f:
                    json.dump(telemetry.to_dict(), f, indent=2)
                print(f"    Saved: {telemetry_path.name}")

                print(f"    Title block location: {telemetry.title_block_location}")
                print(f"    Title block bbox%: {telemetry.title_block_bbox}")
                print(f"    Zones detected: {list(telemetry.zones.keys())}")

                pdf_summary = {
                    'pdf': pdf_name,
                    'total_pages': total_pages,
                    'location': telemetry.title_block_location,
                    'zones': list(telemetry.zones.keys()),
                    'pages_processed': [],
                    'status': 'success'
                }

                for page_num in pages:
                    print(f"\n  Page {page_num}:")
                    page_prefix = f"p{page_num:02d}"

                    try:
                        # Step 2: Render full page
                        print(f"    Step 2: Rendering full page at 150 DPI...")
                        page_image = handler.get_page_image(page_num - 1, dpi=150)
                        page_w, page_h = page_image.size

                        raw_path = pdf_dir / f"{page_prefix}_1_full_page.png"
                        page_image.save(raw_path)
                        print(f"      Saved: {raw_path.name} ({page_w}x{page_h})")

                        # Step 3: Calculate title block bbox in pixels
                        detection = discovery.detect_title_block(page_image)
                        tb_pixels = detection['bbox']
                        tb_bbox = telemetry.title_block_bbox
                        tb_width_pct = (tb_bbox.x2 - tb_bbox.x1) * 100
                        print(f"    Step 3: Title block bbox")
                        print(f"      Percent: x1={tb_bbox.x1:.2f}, y1={tb_bbox.y1:.2f}, x2={tb_bbox.x2:.2f}, y2={tb_bbox.y2:.2f}")
                        print(f"      Width: {tb_width_pct:.1f}% of page")
                        print(f"      Pixels: {tb_pixels}")

                        # Draw title block on full page
                        annotated = draw_labeled_bbox(page_image, tb_pixels, "TITLE_BLOCK", 'red')
                        annotated_path = pdf_dir / f"{page_prefix}_2_with_titleblock_bbox.png"
                        annotated.save(annotated_path)
                        print(f"      Saved: {annotated_path.name}")

                        # Step 4: Crop title block
                        print(f"    Step 4: Cropping title block...")
                        title_block_img = page_image.crop(tb_pixels)
                        tb_w, tb_h = title_block_img.size

                        tb_path = pdf_dir / f"{page_prefix}_3_titleblock_crop.png"
                        title_block_img.save(tb_path)
                        print(f"      Saved: {tb_path.name} ({tb_w}x{tb_h})")

                        # Step 5: Draw ALL zones on title block
                        print(f"    Step 5: Drawing zones on title block...")
                        zone_colors = {
                            'sheet_identification': 'green',
                            'revision_block': 'blue',
                            'project_info': 'orange',
                            'firm_block': 'purple'
                        }

                        tb_annotated = title_block_img.copy()
                        for zone_name, zone in telemetry.zones.items():
                            zb = zone.bbox_percent
                            zone_pixels = (
                                int(zb.x1 * tb_w),
                                int(zb.y1 * tb_h),
                                int(zb.x2 * tb_w),
                                int(zb.y2 * tb_h)
                            )
                            color = zone_colors.get(zone_name, 'gray')
                            tb_annotated = draw_labeled_bbox(tb_annotated, zone_pixels, zone_name, color)
                            print(f"      Zone '{zone_name}': pixels={zone_pixels}, orientation={zone.text_orientation}")

                        zones_path = pdf_dir / f"{page_prefix}_4_titleblock_with_zones.png"
                        tb_annotated.save(zones_path)
                        print(f"      Saved: {zones_path.name}")

                        # Step 6: Crop ONLY the sheet_identification zone
                        print(f"    Step 6: Cropping sheet_identification zone...")
                        if 'sheet_identification' in telemetry.zones:
                            si_zone = telemetry.zones['sheet_identification']
                            si_bbox = si_zone.bbox_percent
                            si_pixels = (
                                int(si_bbox.x1 * tb_w),
                                int(si_bbox.y1 * tb_h),
                                int(si_bbox.x2 * tb_w),
                                int(si_bbox.y2 * tb_h)
                            )

                            si_crop = title_block_img.crop(si_pixels)
                            si_path = pdf_dir / f"{page_prefix}_5_sheet_id_zone_crop.png"
                            si_crop.save(si_path)
                            print(f"      Saved: {si_path.name} ({si_crop.size[0]}x{si_crop.size[1]})")
                            print(f"      THIS is what should be sent to OCR/Vision API!")
                        else:
                            print(f"      WARNING: No sheet_identification zone detected!")

                        pdf_summary['pages_processed'].append(page_num)

                    except Exception as e:
                        print(f"    ERROR on page {page_num}: {e}")
                        import traceback
                        traceback.print_exc()

                summary.append(pdf_summary)

        except Exception as e:
            print(f"  ERROR processing PDF: {e}")
            import traceback
            traceback.print_exc()
            summary.append({
                'pdf': pdf_name,
                'status': 'error',
                'error': str(e)
            })

    # Write summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("VISUAL PROOF COMPLETE")
    print(f"{'='*70}")
    print(f"\nProcessed {len(all_pdfs)} PDFs")
    print(f"Output directory: {run_dir}")
    print(f"Summary: {summary_path}")
    print(f"\nFor each PDF folder, you will find:")
    print(f"  - telemetry.json: Discovery results")
    print(f"  - p01_1_full_page.png: Raw full page")
    print(f"  - p01_2_with_titleblock_bbox.png: Full page with title block highlighted")
    print(f"  - p01_3_titleblock_crop.png: Just the title block")
    print(f"  - p01_4_titleblock_with_zones.png: Title block with zones marked")
    print(f"  - p01_5_sheet_id_zone_crop.png: Just the sheet ID zone")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_visual_proof()
