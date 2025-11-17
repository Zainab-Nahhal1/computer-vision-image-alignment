# Document Image Alignment Using ORB Features and Homography

A compact, production-minded repository demonstrating feature-based image alignment using **ORB** keypoints, **BruteForce-Hamming** matching, and **RANSAC-based homography**. The pipeline is suitable for aligning document photos, perspective correction, and preprocessing for OCR.

## Highlights
- Lightweight: uses ORB (no heavy deep models)
- Robust matching with RANSAC homography
- Simple, reusable `align_images` function
- Demo `main.py` to visualize results
- Tests and CI-friendly structure

## Repo structure
```
cv_alignment_project_full/
│── main.py
│── README.md
│── Makefile
│── requirements.txt
│── .gitignore
│── source/
│   ├── __init__.py
│   └── align_images.py
│── samples/
│   ├── image.jpg
│   └── template.jpg
│── tests/
│   └── test_align.py
```

## Quickstart
1. Create a virtual environment and install dependencies:
```
pip install -r requirements.txt
```

2. Place your images in `samples/` (or use the included examples).

3. Run:
```
python main.py
```

## Notes & Tips
- If `align_images` fails with `Homography could not be computed`, try:
  - Increasing `maxFeatures`
  - Using images with more texture / distinct corners
  - Adjusting `keepPercent` to keep more matches


