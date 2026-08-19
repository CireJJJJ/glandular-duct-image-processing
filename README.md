# Glandular Duct Image Processing

Recovered and cleaned image-processing work from a 2024 InnoViX Lab internship project. The workflow prepares very large glandular images for segmentation by enhancing contrast, splitting images into manageable tiles, extracting contours, and converting reviewed polygons into common annotation formats.

No medical images, labels, or employer datasets are included.

## Pipeline

```text
high-resolution image
        |
        v
grayscale + Gaussian blur + CLAHE
        |
        v
overlapping 1024 x 1024 tiles
        |
        v
Otsu threshold + morphological opening
        |
        v
contour filtering + polygon simplification
        |
        v
LabelMe / YOLO-seg dataset preparation
```

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/glandular_pipeline.py` | Clean command-line pipeline for tiling and contour proposals |
| `notebooks/glandular_duct_workflow.ipynb` | Output-stripped research notebook covering annotation conversion, augmentation, and dataset splitting |
| `requirements.txt` | Minimal dependencies for the command-line pipeline |
| `requirements-notebook.txt` | Additional packages used by the archived notebook |

## Quick start

```bash
python -m venv .venv
python -m pip install -r requirements.txt

python src/glandular_pipeline.py tile data/raw outputs/tiles
python src/glandular_pipeline.py contours outputs/tiles/example outputs/contours
```

The contour output uses LabelMe-style polygon JSON plus binary and overlay images for human review.

## Scope note

The resume also describes an Open3D reconstruction phase. The local source inventory did not contain that implementation, so this repository is intentionally limited to the recovered 2D preprocessing and segmentation-dataset work.
