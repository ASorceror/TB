# Title Block Detection - Comprehensive Plan

## Problem Statement

Current approaches (CV transition detection, AI Vision) are inconsistent:
- CV: tends to find dense structure only (too narrow)
- AI Vision: sometimes wildly wrong (cuts into drawings)
- Neither reliably finds the exact title block boundary

## Research Findings

### Key Paper (2025)
**"Title block detection and information extraction for enhanced building drawings search"**
- arXiv: https://arxiv.org/abs/2504.08645
- Approach: Lightweight CNN detects region → GPT-4o extracts information
- Result: High accuracy on varied drawing types

### Available Libraries

1. **PaddleOCR** - Layout detection + OCR in one package
2. **LayoutParser** - Document layout analysis with Detectron2
3. **YOLOv8** - Fast object detection, minimal training data
4. **OpenCV** - Template matching, feature detection

---

## Proposed Multi-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: COARSE DETECTION                        │
│                                                                     │
│  Multiple methods vote on approximate region (rightmost 15-30%)    │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ CV Edge      │  │ PaddleOCR    │  │ Template     │              │
│  │ Detection    │  │ Layout       │  │ Matching     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         └────────────────┬┴─────────────────┘                       │
│                          ▼                                          │
│                   CONSENSUS VOTE                                    │
│            (take median or most conservative)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: BOUNDARY REFINEMENT                     │
│                                                                     │
│  Crop the coarse region + margin, then refine boundary             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Cropped Region (coarse x1 - 10% to page edge)           │      │
│  │  ┌────────────────────────────────────────────────────┐  │      │
│  │  │                                                    │  │      │
│  │  │   AI Vision analyzes THIS crop only               │  │      │
│  │  │   "Find the left boundary in THIS image"          │  │      │
│  │  │                                                    │  │      │
│  │  └────────────────────────────────────────────────────┘  │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  AI can only adjust within the crop - prevents wild errors         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: STRUCTURE LEARNING                      │
│                                                                     │
│  Learn this PDF's title block template for future pages            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  From 3-5 sample pages, extract:                         │      │
│  │  - Exact bbox (x1, y1, x2, y2)                          │      │
│  │  - Internal zones (firm, project, sheet ID, etc.)       │      │
│  │  - Text orientations                                     │      │
│  │                                                          │      │
│  │  Save as "telemetry" for this PDF                       │      │
│  │  Apply same template to all remaining pages             │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Coarse Detection (Multiple Methods)

### Method A: CV Edge Detection (Current)
```python
# Find consistent edges across sample pages
# Detect transition from zero to non-zero edges
# Returns: x1 estimate (usually 0.86-0.93)
```

### Method B: PaddleOCR Layout Detection
```python
from paddleocr import PPStructure

engine = PPStructure(layout=True, table=False, ocr=False)
result = engine(image_path)

# Find rightmost "table" or "figure" region
# That's likely the title block
```

### Method C: Template Matching (New)
```python
import cv2

# Use a generic title block template (corner with sheet number)
# Find best match position in rightmost 30% of page
template = cv2.imread("title_block_template.png")
result = cv2.matchTemplate(page, template, cv2.TM_CCOEFF_NORMED)
```

### Method D: Line Detection (New)
```python
import cv2

# Detect strong vertical lines using Hough Transform
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200,
                        minLineLength=height*0.5, maxLineGap=20)

# Find vertical lines in rightmost 40% of page
# The leftmost strong vertical line = title block border
```

### Consensus Vote
```python
def get_consensus_x1(cv_x1, paddle_x1, template_x1, lines_x1):
    """
    Combine multiple estimates into final coarse boundary.

    Strategy: Use MEDIAN (robust to outliers)
    Or: Use MAXIMUM (most conservative - smallest crop)
    """
    estimates = [x for x in [cv_x1, paddle_x1, template_x1, lines_x1] if x]

    if len(estimates) >= 2:
        # Median is robust to one outlier
        return np.median(estimates)
    elif estimates:
        return estimates[0]
    else:
        return 0.85  # Safe default
```

---

## Stage 2: Boundary Refinement (Constrained AI Vision)

### Key Insight
AI Vision fails when given the whole page - it sometimes finds drawing content.
**Solution**: Only show AI the CONSTRAINED region where title block MUST be.

```python
def refine_with_ai_vision(page_image, coarse_x1):
    """
    Use AI Vision to refine boundary within a constrained region.
    """
    width, height = page_image.size

    # Create crop: coarse_x1 - 10% to page edge
    # AI can only see this region, preventing wild errors
    crop_x1 = int(max(0, coarse_x1 - 0.10) * width)
    crop_x2 = width

    cropped = page_image.crop((crop_x1, 0, crop_x2, height))

    prompt = """
    This image shows the RIGHT PORTION of a blueprint page.
    The title block LEFT BOUNDARY is somewhere in this image.

    Find the VERTICAL LINE that separates:
    - LEFT: May have some drawing content or whitespace
    - RIGHT: Title block (firm info, project info, sheet number)

    The boundary should:
    - Be at a visible border line OR at natural whitespace
    - NOT cut through any text
    - Include ALL title block content (even sparse vertical text)

    Return the X position as a decimal (0.0 = left edge of THIS CROP, 1.0 = right edge).
    Reply with ONLY a number like: 0.35
    """

    # Call AI Vision with cropped image
    local_x1 = call_ai_vision(cropped, prompt)

    # Convert back to full page coordinates
    global_x1 = (crop_x1 + local_x1 * (crop_x2 - crop_x1)) / width

    return global_x1
```

---

## Stage 3: Structure Learning

### Extract Title Block Template
```python
def learn_title_block_structure(pdf_handler, refined_x1, sample_pages):
    """
    Once we have accurate x1, learn the internal structure.
    """
    title_block_crops = []

    for page_num in sample_pages:
        page_img = pdf_handler.get_page_image(page_num - 1, dpi=150)
        width, height = page_img.size

        # Crop title block
        tb_crop = page_img.crop((int(refined_x1 * width), 0, width, height))
        title_block_crops.append(tb_crop)

    # Use AI Vision to identify zones within title block
    prompt = """
    Analyze this title block crop from a blueprint.

    Identify these zones (as percent of title block dimensions):
    1. firm_info: Company name, logo, contact info
    2. project_info: Project name, address
    3. sheet_identification: Sheet number (A-1, S-1, etc.), sheet title
    4. revision_block: Revision history table
    5. dates_approvals: Date, drawn by, checked by, scale

    For each zone, provide:
    - bbox: {x1, y1, x2, y2} as percentages (0.0 to 1.0)
    - text_orientation: "horizontal" or "vertical"

    Return JSON format.
    """

    zones = call_ai_vision(title_block_crops[0], prompt)

    return {
        'title_block_x1': refined_x1,
        'zones': zones,
        'confidence': 0.95
    }
```

---

## Implementation Roadmap

### Phase 1: Add PaddleOCR Layout Detection (1-2 days)
```bash
pip install paddleocr paddlepaddle
```
- Integrate PP-Structure for layout detection
- Test on 17 PDFs
- Compare with current CV approach

### Phase 2: Add Line Detection (1 day)
- Use OpenCV Hough Transform
- Find strong vertical lines in rightmost region
- Add to consensus vote

### Phase 3: Implement Constrained AI Refinement (1 day)
- Modify AI Vision to only see cropped region
- Test accuracy improvement

### Phase 4: Consensus Pipeline (1 day)
- Combine all methods
- Implement voting/median logic
- Full pipeline test

### Phase 5: Structure Learning (2 days)
- Save learned templates per PDF
- Apply to remaining pages
- Cache for future processing

---

## Alternative Approaches (If Pipeline Fails)

### Option A: Train Custom YOLOv8 Model
1. Annotate 50-100 title blocks using CVAT or Roboflow
2. Train YOLOv8-nano (fast) or YOLOv8-medium (accurate)
3. Deploy as part of pipeline

```python
from ultralytics import YOLO

# Train
model = YOLO('yolov8n.pt')
model.train(data='title_blocks.yaml', epochs=50)

# Inference
results = model('blueprint.png')
title_block_bbox = results[0].boxes[0]  # First detection
```

### Option B: Use Werk24 API (Commercial)
- Specialized for technical drawings
- May be worth the cost for production use
- https://werk24.io/

### Option C: Fine-tune LayoutParser
1. Use existing PubLayNet model as base
2. Fine-tune on blueprint title blocks
3. More accurate than generic layout detection

---

## Success Metrics

| Metric | Target |
|--------|--------|
| x1 accuracy | Within 2% of actual boundary |
| No text cutoff | 100% (never cut through text) |
| No drawing content | 100% (never include drawing tables) |
| Processing time | < 5 seconds per PDF (first page) |
| Consistency | Same result on repeated runs |

---

## Files to Create

```
blueprint_processor/
├── core/
│   ├── title_block_detector.py      # Main pipeline
│   ├── coarse_detection/
│   │   ├── cv_edge_detector.py      # Current CV approach
│   │   ├── paddle_layout.py         # PaddleOCR integration
│   │   ├── line_detector.py         # Hough line detection
│   │   └── template_matcher.py      # Template matching
│   ├── boundary_refiner.py          # Constrained AI Vision
│   └── structure_learner.py         # Zone detection
├── templates/
│   └── title_block_corner.png       # Generic template
└── tests/
    └── test_title_block_detection.py
```

---

## Next Steps

1. **Immediate**: Install PaddleOCR and test layout detection
2. **This week**: Implement multi-method consensus pipeline
3. **Next week**: Add constrained AI refinement
4. **Evaluate**: If accuracy < 95%, consider training custom model

---

## References

- [arXiv: Title block detection (2025)](https://arxiv.org/abs/2504.08645)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [LayoutParser GitHub](https://github.com/Layout-Parser/layout-parser)
- [Detectron2 GitHub](https://github.com/facebookresearch/detectron2)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Architectural Blueprint](https://universe.roboflow.com/estima-zza1m/architectural-blueprint)
- [Werk24 Technical Drawing API](https://werk24.io/)
