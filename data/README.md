# Dry Bean Dataset

## Source

The Dry Bean Dataset is from the **UCI Machine Learning Repository**:
- **URL:** https://archive.ics.uci.edu/dataset/602/dry-bean-dataset
- **Citation:** Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## Download Instructions

The dataset is automatically downloaded when you run the training pipeline.
You can also download it manually:

```bash
python src/data_preprocessing.py
```

This will download the dataset as `dry_beans.csv` in this directory.

## Dataset Description

- **Samples:** 13,611
- **Features:** 16 (geometric/shape measurements from grain images)
- **Target:** 7 dry bean varieties (SEKER, BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SIRA)
- **Task:** Multi-class classification

### Features

| Feature | Description |
|---------|-------------|
| Area | The area of the bean zone in pixels |
| Perimeter | The length of the bean's border in pixels |
| MajorAxisLength | The longest line distance within the bean |
| MinorAxisLength | The shortest line distance within the bean |
| AspectRatio | Ratio of MajorAxisLength to MinorAxisLength |
| Eccentricity | Eccentricity of the equivalent ellipse |
| ConvexArea | Number of pixels in the smallest convex polygon containing the bean |
| EquivDiameter | Diameter of a circle with the same area as the bean |
| Extent | Ratio of bean pixels to bounding rectangle pixels |
| Solidity | Ratio of bean pixels to convex hull pixels |
| Roundness | Calculated as (4 * π * Area) / Perimeter² |
| Compactness | Calculated as EquivDiameter / MajorAxisLength |
| ShapeFactor1 | Shape factor 1 |
| ShapeFactor2 | Shape factor 2 |
| ShapeFactor3 | Shape factor 3 |
| ShapeFactor4 | Shape factor 4 |
