"""
Blueprint Processor V4.2.1 - Coordinate Conversion Helpers
Handles conversion between pixels (image coordinates) and points (PDF coordinates).
"""

from typing import List, Tuple, Union

# Standard DPI used for rendering
DEFAULT_DPI = 200

# PDF standard is 72 points per inch
POINTS_PER_INCH = 72


def pixels_to_points(bbox_pixels: Union[List[float], Tuple[float, ...]], dpi: int = DEFAULT_DPI) -> List[float]:
    """
    Convert pixel coordinates to PDF points.

    RegionDetector returns bbox in pixels from rendered image at DPI.
    PyMuPDF uses points (72 points per inch).

    Args:
        bbox_pixels: Bounding box in pixels [x0, y0, x1, y1]
        dpi: DPI used for rendering (default 200)

    Returns:
        Bounding box in points [x0, y0, x1, y1]
    """
    scale = POINTS_PER_INCH / dpi  # 72/200 = 0.36
    return [coord * scale for coord in bbox_pixels]


def points_to_pixels(bbox_points: Union[List[float], Tuple[float, ...]], dpi: int = DEFAULT_DPI) -> List[float]:
    """
    Convert PDF points to pixel coordinates.

    Args:
        bbox_points: Bounding box in points [x0, y0, x1, y1]
        dpi: DPI used for rendering (default 200)

    Returns:
        Bounding box in pixels [x0, y0, x1, y1]
    """
    scale = dpi / POINTS_PER_INCH  # 200/72 = 2.778
    return [coord * scale for coord in bbox_points]


def is_point_in_bbox(point: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point is inside a bounding box.

    Args:
        point: (x, y) coordinates
        bbox: (x0, y0, x1, y1) bounding box

    Returns:
        True if point is inside bbox
    """
    x, y = point
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def bbox_overlap(bbox1: Tuple[float, float, float, float],
                 bbox2: Tuple[float, float, float, float]) -> bool:
    """
    Check if two bounding boxes overlap.

    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)

    Returns:
        True if boxes overlap
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # Check for no overlap conditions
    if x1_1 < x0_2 or x1_2 < x0_1:
        return False
    if y1_1 < y0_2 or y1_2 < y0_1:
        return False

    return True


def bbox_intersection(bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Calculate the intersection of two bounding boxes.

    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)

    Returns:
        Intersection bounding box, or (0, 0, 0, 0) if no overlap
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    x0 = max(x0_1, x0_2)
    y0 = max(y0_1, y0_2)
    x1 = min(x1_1, x1_2)
    y1 = min(y1_1, y1_2)

    if x0 < x1 and y0 < y1:
        return (x0, y0, x1, y1)
    return (0, 0, 0, 0)


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: Bounding box (x0, y0, x1, y1)

    Returns:
        Center point (x, y)
    """
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate the area of a bounding box.

    Args:
        bbox: Bounding box (x0, y0, x1, y1)

    Returns:
        Area in square units
    """
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def get_vertical_zone(y: float, bbox: Tuple[float, float, float, float]) -> str:
    """
    Determine which vertical zone a y-coordinate falls into within a bounding box.

    Zones:
    - header: top 15% (company logos, etc.)
    - title: 15% to 65% (where sheet title usually is)
    - info: bottom 35% (labels, sheet number, etc.)

    Args:
        y: Y-coordinate in same units as bbox
        bbox: Container bounding box (x0, y0, x1, y1)

    Returns:
        Zone name: 'header', 'title', or 'info'
    """
    _, y0, _, y1 = bbox
    height = y1 - y0

    if height <= 0:
        return 'info'

    # Calculate relative position (0 = top, 1 = bottom)
    relative_y = (y - y0) / height

    if relative_y < 0.15:
        return 'header'
    elif relative_y < 0.65:
        return 'title'
    else:
        return 'info'


def get_title_zone_bbox(title_block_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Calculate the title zone bounding box within a title block.

    Title zone is 15% to 65% of the vertical extent.

    Args:
        title_block_bbox: Full title block bounding box (x0, y0, x1, y1)

    Returns:
        Title zone bounding box (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = title_block_bbox
    height = y1 - y0

    title_y0 = y0 + (height * 0.15)
    title_y1 = y0 + (height * 0.65)

    return (x0, title_y0, x1, title_y1)
