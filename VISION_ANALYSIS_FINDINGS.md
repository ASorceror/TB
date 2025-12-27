# Vision Analysis Findings - V5.0 Title Block Patterns

## Accuracy Test Results
- **Sheet Number Accuracy: 40.2%** (979/2435)
- **Sheet Title Accuracy: 43.3%** (1055/2435)
- Processing time: 8+ hours for 17 PDFs

## Critical Issue Found

### Preprocessing Inconsistency
The `region_detector.py` is using AGGRESSIVE preprocessing (Otsu binarization) while `extractor.py` was updated to use LSTM-friendly preprocessing. This inconsistency causes:

1. **region_detector.py (line 289-298)**: Uses `apply_threshold=True` with `threshold_method='otsu'`
2. **extractor.py**: Uses `preprocessing_mode='lstm'` (skips binarization)

**Fix Required**: Update `region_detector.py` to use LSTM-friendly preprocessing.

---

## PDF Title Block Pattern Analysis

### Pattern 1: Kriser's Natural Pet (Right Edge Vertical)
**Files**: `3-7-25 Kriser's Highand Final Set.pdf`
**Extraction Result**: Empty ('') for most pages
**Ground Truth**: T2, A0, A0.1, A0.2, A2, etc.

**Title Block Layout**:
- Position: RIGHT edge of page (vertical strip)
- Vertical project text: "KRISER'S NATURAL PET"
- Address below project name (vertical)
- Legend/keyed notes section
- **Sheet Title**: Box near bottom with drawing name (e.g., "DEMOLITION FLOOR PLAN")
- **Sheet Number**: Small box at very bottom right (e.g., "A0")

**Why Extraction Failed**:
- Title block is a full-height RIGHT STRIP, not bottom-right corner
- Sheet number label "SHEET NUMBER" is present but may not be detected
- The extraction strategies rely on finding labels first

---

### Pattern 2: Janesville Nissan (Right Edge Vertical)
**Files**: `Janesville Nissan Full set Issued for Bids.pdf`
**Extraction Result**: Empty ('') for most pages
**Ground Truth**: A-100, A-1, D-1, A-2.1, etc.

**Title Block Layout**:
- Position: RIGHT edge (vertical strip similar to Kriser's)
- Project: "BUILDING ALTERATIONS, JANESVILLE NISSAN"
- Contains drawing index on cover sheet
- **Sheet Title**: In title block section
- **Sheet Number**: Small box at bottom (e.g., "A-1", "D-1")

**Key Observation**:
- Sheet numbers clearly visible in images
- Uses dash format (A-1 instead of A1.0)
- Same vertical strip pattern as Kriser's

---

### Pattern 3: Senju Office TI (Traditional Bottom-Right)
**Files**: `Senju Office TI Arch Permit Set 2.4.20.pdf`
**Extraction Result**: Got actual sheet numbers (A2.0) but ground truth expected drawing references (A15)

**Title Block Layout**:
- Position: Traditional BOTTOM-RIGHT corner
- "PEAK" Construction logo
- Schedules/tables in title block area
- **Sheet Title**: "Code Review, Schedules and ADA Requirements"
- **Sheet Number**: "A1.0" in bottom-right title block box

**Ground Truth Issue**:
- Ground truth file may have WRONG values
- "A15" in bottom-left is a DRAWING INDEX REFERENCE, not sheet number
- Actual sheet number is "A1.0" from title block

---

### Pattern 4: Chiro One Wellness (Traditional Right Edge)
**Files**: `0_full_permit_set_chiro_one_evergreen_park.pdf`
**Likely Working**: This appears to be a cleaner format

**Title Block Layout**:
- Position: RIGHT edge (traditional layout)
- Project name vertical: "CHIRO ONE WELLNESS CENTER"
- Address: "95TH STREET, EVERGREEN PARK, CHICAGO, ILLINOIS"
- Clear "DOOR SCHEDULE" table
- **Sheet Number**: "A-2" clearly visible in title block

---

## Root Cause Analysis

### 1. Title Block Region Detection
The region detector searches these areas:
- `bottom_right_tight` (primary)
- `bottom_right`
- `right_strip_bottom`
- `right_strip` (full height)

For vertical title blocks (Kriser's, Janesville), `right_strip` should be used, but:
- It may not be selected if keywords aren't found due to OCR issues
- The default fallback is `bottom_right_tight` which misses vertical layouts

### 2. OCR Preprocessing Mismatch
- Region detector uses aggressive Otsu binarization
- This can damage thin characters in small title block text
- V5.0 LSTM mode was added to extractor but not region detector

### 3. Label-First Strategy Limitation
The extractor only finds sheet numbers near labels like:
- "SHEET NUMBER", "SHEET NO", "SHEET #"
- "DWG NO", "DRAWING NO"

If OCR doesn't capture these labels clearly, no sheet number is found.

### 4. Ground Truth Accuracy
Some ground truth values appear incorrect:
- Senju PDF: Expected "A15" but that's a drawing reference, not sheet number
- Need to verify ground truth accuracy

---

## Recommended Fixes

### Fix 1: Update region_detector.py preprocessing
```python
# Change from:
processed_image = preprocess_for_ocr(
    image,
    apply_grayscale=True,
    apply_denoise=True,
    apply_threshold=True,  # <-- REMOVE
    threshold_method='otsu',  # <-- REMOVE
    ...
)

# To:
processed_image = preprocess_for_ocr(
    image,
    apply_grayscale=True,
    apply_denoise=True,
    apply_border=True,
    border_size=10,
    invert_if_light_text=True,
    preprocessing_mode='lstm',  # <-- ADD
)
```

### Fix 2: Add fallback strategies for sheet number
- If label-based extraction fails, try pattern matching in title block region only
- Look for isolated alphanumeric codes in bottom-right corner of title block
- Use spatial analysis to find text in "sheet number box" position

### Fix 3: Improve right-strip detection
- Increase scoring for right_strip when keywords are found there
- Consider aspect ratio of detected title block region

### Fix 4: Validate and clean ground truth
- Cross-reference ground truth values with actual sheet numbers
- Distinguish between drawing index references and actual sheet numbers

---

## V5.0 Implementation Results

### Fixes Applied

1. **region_detector.py**: Updated to use LSTM-friendly preprocessing (no binarization)
2. **extractor.py**: Added Fallback 5 - Outline Font OCR with:
   - Heavy morphological dilation (5x5 kernel, 2 iterations)
   - 2x upscaling for better OCR
   - PSM 6 mode (block of text)
   - O/Q to 0 post-processing fix
   - I to 1 post-processing fix
   - Smart prefix detection for sheet numbers

### Test Results After V5.0 Fixes

**Kriser's Natural Pet PDF:**
- Page 2: **A0** extracted (via outline_font_ocr)
- Page 3: **A0** extracted (via outline_font_ocr)
- Page 4: Still empty (partial OCR result "A")

**Janesville Nissan PDF:**
- Pages 1-3: Still empty
- Root cause: Horizontal title block layout - outline font OCR designed for vertical strips
- The sheet number box location is different in horizontal layouts

### Key Technical Findings

1. **Thin outline fonts** in architectural drawings are the main blocker for OCR accuracy
2. **Character whitelist** in pytesseract can cause empty results on Windows
3. **O/Q confusion** is common (0 read as O or Q)
4. **Morphological dilation** (erosion on binary image) effectively thickens thin strokes
5. **Sheet numbers are NOT in PDF text layer** - they're rendered as vector paths, requiring OCR
6. **Title block layouts vary significantly** between architectural firms:
   - Vertical right-edge strip (Kriser's)
   - Horizontal bottom strip (Janesville)
   - Traditional bottom-right corner (Senju)

### Remaining Work

1. Adapt outline font OCR for horizontal title block layouts
2. Improve partial OCR recovery (e.g., "A" -> try to find rest of number)
3. Validate ground truth accuracy (some values may be drawing references vs actual sheet numbers)
