"""
Coarse Detection Methods for Title Block Detection

Multiple methods to estimate the approximate title block region:
- CV Edge Transition Detection
- Hough Line Detection
- PaddleOCR Layout (if effective)
- Template Matching (future)
"""

from .line_detector import (
    find_vertical_lines,
    cluster_lines_by_x,
    detect_title_block_border,
    visualize_lines
)

from .cv_transition import (
    get_common_edges,
    get_majority_edges,
    find_transition_x1,
    detect_title_block_cv,
    visualize_edges
)

from .consensus import (
    CoarseDetector,
    get_consensus_x1
)

__all__ = [
    # Line detector
    'find_vertical_lines',
    'cluster_lines_by_x',
    'detect_title_block_border',
    'visualize_lines',
    # CV transition
    'get_common_edges',
    'get_majority_edges',
    'find_transition_x1',
    'detect_title_block_cv',
    'visualize_edges',
    # Consensus
    'CoarseDetector',
    'get_consensus_x1'
]
