# Major Question: Can I Predict Whether a Person Has Ever Had a Stroke?

This project explores whether it is possible to use demographic, lifestyle, and nutritional factors to **predict the likelihood that a person has experienced a stroke**, using Support Vector Machine (SVM) models with linear, polynomial, and radial basis function (RBF) kernels.

---

## Target Variable: `STROKEV`

Indicates whether the adult respondent has *ever* been told by a healthcare professional that they had a stroke.

| Value | Meaning                         |
|-------|----------------------------------|
| 1     | No (Did not have a stroke)      |
| 2     | Yes (Had a stroke)              |
| 0     | Not in Universe (NIU)           |
| 7     | Unknown - Refused               |
| 8     | Unknown - Not Ascertained       |
| 9     | Unknown - Don't Know            |

**Note**: For classification purposes, I have:
- Encoded `STROKEV = 2` as `1` (stroke history)
- Encoded `STROKEV = 1` as `0` (no stroke)
- Excluded entries with values {0, 7, 8, 9}

---

## Feature Descriptions

Below are the selected features used to predict stroke history:

| Feature        | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `BMICALC`      | Body Mass Index (BMI), calculated from height and weight                   |
| `HRSLEEP`      | Average number of hours of sleep per night                                 |
| `SODAPNO`      | Weekly consumption of soda                                                  |
| `VEGENO`       | Weekly consumption of vegetables                                            |
| `MOD10DMIN`    | Minutes of moderate physical activity per day (≥10 minutes/session)         |
| `PIZZANO`      | Weekly consumption of pizza                                                 |

---

## Preprocessing Guidelines

- Drop rows where `STROKEV` ∈ {0, 7, 8, 9}
- Map target:
  - `STROKEV = 2` → `1` (had a stroke)
  - `STROKEV = 1` → `0` (did not have a stroke)
- Handle missing/invalid codes (e.g., 996, 997, 998, 999)
- Standardize continuous features before applying SVM
