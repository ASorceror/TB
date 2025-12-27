"""
Hough Line Detection for Title Block Borders

Uses OpenCV's HoughLinesP to find strong vertical lines in the rightmost
portion of blueprint pages. The leftmost consistent vertical line in this
region likely represents the title block left border.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import cv2


def find_vertical_lines(
    page_image: Image.Image,
    search_region: Tuple[float, float] = (0.60, 1.0),
    min_line_length_ratio: float = 0.3,
    max_line_gap: int = 30,
    angle_tolerance: float = 5.0,
    threshold: int = 100
) -> List[Dict]:
    """
    Find strong vertical lines in the specified region of the page.

    Args:
        page_image: PIL Image of the page
        search_region: (start, end) as fraction of page width to search
        min_line_length_ratio: Minimum line length as fraction of page height
        max_line_gap: Maximum gap between line segments to join
        angle_tolerance: Degrees from vertical to still consider "vertical"
        threshold: Accumulator threshold for HoughLinesP

    Returns:
        List of detected vertical lines with their properties
    """
    width, height = page_image.size

    # Convert to grayscale numpy array
    gray = np.array(page_image.convert('L'))

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Mask to only search in the specified region
    region_start = int(search_region[0] * width)
    region_end = int(search_region[1] * width)

    mask = np.zeros_like(edges)
    mask[:, region_start:region_end] = 255
    edges_masked = cv2.bitwise_and(edges, mask)

    # Detect lines using probabilistic Hough Transform
    min_line_length = int(height * min_line_length_ratio)

    lines = cv2.HoughLinesP(
        edges_masked,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle from vertical
            if x2 - x1 == 0:
                angle = 0  # Perfectly vertical
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
                angle = 90 - angle  # Convert to angle from vertical

            # Only keep lines close to vertical
            if angle <= angle_tolerance:
                # Calculate average x position
                x_avg = (x1 + x2) / 2
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                vertical_lines.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'x_avg': x_avg,
                    'x_avg_pct': x_avg / width,
                    'length': line_length,
                    'length_pct': line_length / height,
                    'angle_from_vertical': angle
                })

    # Sort by x position (leftmost first)
    vertical_lines.sort(key=lambda l: l['x_avg'])

    return vertical_lines


def cluster_lines_by_x(
    lines: List[Dict],
    cluster_threshold: float = 0.02
) -> List[Dict]:
    """
    Cluster nearby vertical lines into groups.

    Args:
        lines: List of detected lines
        cluster_threshold: Maximum x distance (as fraction) to cluster

    Returns:
        List of clusters with aggregated properties
    """
    if not lines:
        return []

    clusters = []
    current_cluster = [lines[0]]

    for line in lines[1:]:
        # Check if this line is close to the current cluster
        cluster_x = np.mean([l['x_avg_pct'] for l in current_cluster])

        if abs(line['x_avg_pct'] - cluster_x) <= cluster_threshold:
            current_cluster.append(line)
        else:
            # Finalize current cluster and start new one
            clusters.append(_summarize_cluster(current_cluster))
            current_cluster = [line]

    # Don't forget the last cluster
    if current_cluster:
        clusters.append(_summarize_cluster(current_cluster))

    return clusters


def _summarize_cluster(lines: List[Dict]) -> Dict:
    """Summarize a cluster of lines into a single line descriptor."""
    x_positions = [l['x_avg_pct'] for l in lines]
    lengths = [l['length_pct'] for l in lines]

    return {
        'x_avg_pct': np.mean(x_positions),
        'x_min_pct': min(x_positions),
        'x_max_pct': max(x_positions),
        'total_length_pct': sum(lengths),
        'max_length_pct': max(lengths),
        'num_segments': len(lines),
        'lines': lines
    }


def detect_title_block_border(
    page_images: List[Image.Image],
    search_region: Tuple[float, float] = (0.60, 0.98),
    min_agreement: int = 2
) -> Optional[float]:
    """
    Find the title block left border using Hough line detection across multiple pages.

    Args:
        page_images: List of PIL Images (sample pages from PDF)
        search_region: Region to search for vertical lines
        min_agreement: Minimum number of pages that must have a line at similar position

    Returns:
        x1 position as fraction (0.0-1.0) or None if not found
    """
    all_line_positions = []

    for img in page_images:
        lines = find_vertical_lines(img, search_region=search_region)
        clusters = cluster_lines_by_x(lines)

        # Get x positions of strong clusters (total length > 30% of page)
        for cluster in clusters:
            if cluster['total_length_pct'] > 0.3:
                all_line_positions.append(cluster['x_avg_pct'])

    if not all_line_positions:
        return None

    # Cluster line positions across pages
    positions = np.array(sorted(all_line_positions))

    # Find positions that appear in multiple pages
    # Use a simple binning approach
    bins = np.arange(search_region[0], search_region[1] + 0.02, 0.02)
    hist, bin_edges = np.histogram(positions, bins=bins)

    # Find bins with enough agreement
    consistent_positions = []
    for i, count in enumerate(hist):
        if count >= min_agreement:
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            consistent_positions.append((bin_center, count))

    if not consistent_positions:
        return None

    # Return the leftmost consistent position (likely title block border)
    consistent_positions.sort(key=lambda x: x[0])
    return consistent_positions[0][0]


def visualize_lines(
    page_image: Image.Image,
    lines: List[Dict],
    output_path: Optional[Path] = None
) -> Image.Image:
    """
    Draw detected lines on the image for visualization.
    """
    result = page_image.convert('RGB')
    draw = ImageDraw.Draw(result)

    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']

    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        draw.line(
            [(line['x1'], line['y1']), (line['x2'], line['y2'])],
            fill=color,
            width=3
        )
        # Label with x position
        draw.text(
            (line['x1'], line['y1'] - 20),
            f"x={line['x_avg_pct']:.3f}",
            fill=color
        )

    if output_path:
        result.save(output_path)

    return result
