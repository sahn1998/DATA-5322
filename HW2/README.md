# Major Question: Can I Predict Whether a Person Has Diabetes?

This project explores whether it is possible to use demographic, lifestyle, and nutritional factors to **predict the likelihood that a person will have diabetes**, using Support Vector Machine (SVM) models with linear, polynomial, and radial basis function (RBF) kernels.

---

## Target Variable: `DIABETICEV`

Indicates whether the adult respondent had ever been diagnosed with "diabetes or sugar diabetes" by a doctor or other health professional.

| Value | Meaning                         |
|-------|---------------------------------|
| 1     | No                              |
| 2     | Yes (Diabetes)                  |
| 3     | Borderline                      |
| 7     | Unknown - Refused               |
| 8     | Unknown - Not Ascertained       |
| 9     | Unknown - Don't Know            |

**Note**: For classification purposes, I have:
- Encoded `DIABETICEV = 3` as `1` (Borderline: If someone is borderline diabetes, I assume that it means that they will most likely have diabetes in the near future.)
- Encoded `DIABETICEV = 2` as `1` (Diabetes history)
- Encoded `DIABETICEV = 1` as `0` (no diabetes)
- Excluded entries with values {0, 7, 8, 9}
---

## Feature Descriptions

Below are the selected features used to predict diabetes history:

| Feature        | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `BMICALC`      | Body Mass Index (BMI), calculated from height and weight                    |
| `SODAPNO`      | Weekly consumption of soda                                                  |
| `SALADSNO`     | Weekly consumption of salad                                                 |
| `FRIESPNO`     | Weekly consumption of fries                                                 |
| `VEGENO`       | Weekly consumption of vegetables                                            |
| `MOD10DMIN`    | Minutes of moderate physical activity per day (≥10 minutes/session)         |
| `PIZZANO`      | Weekly consumption of pizza                                                 |

---

## Preprocessing Guidelines

- Drop rows where `DIABETICEV` ∈ {0, 7, 8, 9}
- Map target:
  - `DIABETICEV = 3` → `1` (had/will have diabetes)
  - `DIABETICEV = 2` → `1` (had/will have diabetes)
  - `DIABETICEV = 1` → `0` (no diabetes)
- Handle missing/invalid codes (e.g., 996, 997, 998, 999)
- Standardize continuous features before applying SVM
