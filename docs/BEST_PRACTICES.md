# Blueprint Processor - Tool Best Practices Reference

This document consolidates best practices for all major libraries used in the Blueprint Processor codebase. Use this as a reference when developing or maintaining the extraction pipeline.

---

## Table of Contents

1. [Tesseract OCR](#tesseract-ocr)
2. [PyMuPDF (fitz)](#pymupdf-fitz)
3. [PIL/Pillow](#pilpillow)
4. [Quick Reference Tables](#quick-reference-tables)

---

## Tesseract OCR

### Optimal DPI Settings

| Document Type | Recommended DPI | Notes |
|---------------|-----------------|-------|
| Standard documents | 300 | Industry standard minimum |
| Blueprints/Technical drawings | 300 | PSM 11 recommended |
| Fine print | 400-600 | For small text |
| Speed-critical | 150-200 | Trade-off accuracy |

**Critical metric**: Character height should be **25-40 pixels** for optimal accuracy. Above 90 pixels, accuracy can degrade.

### PSM Mode Selection

| PSM | Use Case |
|-----|----------|
| 3 (Default) | General documents, mixed layouts |
| 6 | Single uniform block of text |
| 7 | Single text line |
| 11 | **Sparse text - best for blueprints/forms** |

**For blueprints**: Start with PSM 11 for scattered labels. Fall back to PSM 3 if needed.

### OEM Mode Selection

| OEM | Engine | Best For |
|-----|--------|----------|
| 0 | Legacy | Speed, simple documents |
| 1 | LSTM | Accuracy, modern documents |
| 3 | Auto | General use (recommended) |

### Preprocessing Pipeline (Required Order)

1. **Grayscale conversion** - Always first
2. **Noise removal** - MedianBlur(3) for salt-and-pepper
3. **Binarization** - Otsu's thresholding
4. **Deskewing** - Critical for scanned documents
5. **Border padding** - Add ~10px white border

### Configuration for Technical Drawings

```python
TESSERACT_CONFIG = {
    'page': '--oem 3 --psm 11',
    'title_block': '--oem 1 --psm 6',
    'sheet_number': '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.',
}
```

### Common Pitfalls

1. **Wrong PSM mode** - Use PSM 11 for blueprints, not PSM 3
2. **Skewed pages** - Always deskew before OCR
3. **High DPI causing issues** - 200-300 DPI often works better than 400+
4. **Digit confusion** - OCR often confuses 4↔7, 1↔l, 0↔O
5. **Missing borders** - Add padding to prevent "empty page" errors

---

## PyMuPDF (fitz)

### DPI/Matrix Settings

```python
# Standard approach
zoom = desired_dpi / 72  # PDF standard is 72 DPI
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat)

# Direct DPI (PyMuPDF 1.19.2+)
pix = page.get_pixmap(dpi=300)
```

| Use Case | DPI | Notes |
|----------|-----|-------|
| Screen display | 96 | Fastest |
| OCR preparation | 300 | Standard for text recognition |
| High quality | 300+ | Use clips for large pages |

### Memory Management

```python
# Clean cache periodically for large PDFs
import fitz
fitz.Tools.reduce_store(100)  # Empty store

# Process one page at a time
for page_num in range(doc.page_count):
    page = doc[page_num]
    # Process...
    if page_num % 50 == 0:
        fitz.Tools.reduce_store(100)
```

### Text Extraction Methods

| Method | Speed | Use Case |
|--------|-------|----------|
| `get_text()` | Fastest | Simple text extraction |
| `get_text("blocks")` | Fast | Layout preservation |
| `get_text("dict")` | Moderate | Structured data with coordinates |

### Image Extraction

```python
# Fast: Extract embedded image bytes
image_dict = doc.extract_image(xref)
image_bytes = image_dict["image"]

# Render page as image (for vector content)
pix = page.get_pixmap(dpi=300)

# Convert to PIL
from PIL import Image
pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
```

### Detecting Scanned PDFs

```python
def is_scanned_pdf(page):
    """Check if page is scanned (image-based)"""
    images = page.get_images()
    text = page.get_text().strip()

    if images and not text:
        return True
    if images:
        img_rect = page.get_image_rects(images[0])[0]
        coverage = img_rect.get_area() / page.rect.get_area()
        if coverage > 0.8:
            return True
    return False
```

### Common Pitfalls

1. **Memory leaks** - Clean store periodically
2. **Coordinate system** - Check page rotation and CropBox
3. **Missing text** - Scanned PDFs need OCR
4. **EPUB performance** - Don't jump to absolute page numbers

---

## PIL/Pillow

### OCR Preprocessing

```python
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

def preprocess_for_ocr(image):
    """Standard OCR preprocessing pipeline"""
    # 1. Grayscale
    if image.mode != 'L':
        image = ImageOps.grayscale(image)

    # 2. Noise removal
    image = image.filter(ImageFilter.MedianFilter(size=3))

    # 3. Contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    return image
```

### Resizing Algorithms

| Algorithm | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| NEAREST | Fastest | Lowest | Solid colors, speed-critical |
| BILINEAR | Fast | Medium | Quick thumbnails |
| BICUBIC | Medium | High | General use (default) |
| LANCZOS | Slowest | Highest | Quality-critical, downscaling |

```python
# For OCR: Use LANCZOS when downscaling
resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

# For speed: Use thumbnail() - modifies in-place
image.thumbnail((max_w, max_h), Image.Resampling.BICUBIC)
```

### Memory Management

```python
# Use context managers
with Image.open(path) as img:
    result = img.resize((400, 300))
    result.save(output_path)

# For large images: Process in chunks
def process_in_chunks(image, chunk_size=512):
    width, height = image.size
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            box = (x, y, min(x+chunk_size, width), min(y+chunk_size, height))
            chunk = image.crop(box)
            yield process_chunk(chunk)
```

### Color Mode Handling

```python
def ensure_rgb(image):
    """Safely convert to RGB"""
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode != 'RGB':
        return image.convert('RGB')
    return image
```

### Common Pitfalls

1. **Memory leaks** - Always close images or use context managers
2. **Aspect ratio** - Use `thumbnail()` to preserve proportions
3. **Mode conversion** - Check mode before converting
4. **Rotation truncation** - Use `expand=True` in `rotate()`
5. **Filter compatibility** - Convert to RGB before applying filters

---

## Quick Reference Tables

### DPI Selection Guide

| Scenario | Tesseract | PyMuPDF | Notes |
|----------|-----------|---------|-------|
| Blueprint full page | 300 | 300 | Standard OCR quality |
| Title block region | 300 | 300-400 | Small text may need higher |
| Speed-critical | 150-200 | 150 | Trade-off accuracy |
| Problem pages | 150 | 150 | Try lower DPI if orientation detection fails |

### Preprocessing Pipeline Order

1. Load image (PyMuPDF → PIL)
2. Grayscale conversion (PIL)
3. Noise removal (PIL MedianFilter)
4. Contrast enhancement (PIL ImageEnhance)
5. Binarization (PIL or OpenCV Otsu)
6. Deskewing (if needed)
7. OCR (Tesseract)

### Error Recovery Strategies

| Issue | Solution |
|-------|----------|
| Empty OCR result | Try lower DPI (150), different PSM |
| Garbled text | Check orientation, try deskewing |
| Wrong digits (4↔7) | Use Vision API verification |
| Memory issues | Process one page at a time, clean store |
| Slow processing | Use PSM 11 instead of PSM 3 for blueprints |

---

## Version History

- **2024-12-25**: Initial documentation compiled from research
