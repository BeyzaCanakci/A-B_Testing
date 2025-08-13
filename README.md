# A/B Testing: Comparing Conversion Rates of Bidding Methods

This repository contains a Python script for conducting an **A/B test analysis** of two bidding strategies ("Maximum Bidding" vs "Average Bidding") for the client **bombabomba.com**. The goal is to determine whether the new "Average Bidding" method leads to more conversions (purchases) than the traditional "Maximum Bidding" method.

---

## Business Problem

Facebook has introduced a new bidding type called *average bidding* as an alternative to *maximum bidding*. The client wants to test if the new method increases conversions.

- **Key Metric:** Purchase (number of products purchased)
- **Control Group:** Maximum Bidding
- **Test Group:** Average Bidding

---

## Dataset

- Data is stored in an Excel file: `Dataset/ab_testing.xlsx`
- There are two sheets: `Control Group` and `Test Group`
- **Columns:**
  - `impression`: Number of ad impressions
  - `Click`: Number of clicks on the ads
  - `Purchase`: Number of products purchased after clicks
  - `Earning`: Revenue earned after purchases

---

## Analysis Workflow

1. **Data Preparation & Exploration**
   - Load data from Excel sheets
   - Inspect dataframes for size, types, nulls, and statistics
   - Merge both groups into a single dataframe

2. **Hypothesis Definition**
   - **Null Hypothesis (H0):** No difference in mean Purchase between groups
   - **Alternative Hypothesis (H1):** Difference exists in mean Purchase

3. **Statistical Testing**
   - **Normality Test:** Shapiro-Wilk test
   - **Variance Homogeneity:** Levene's test
   - **Statistical Test Used:** Independent Two-Sample t-Test

4. **Results Analysis**
   - Report means, standard deviations, and 95% confidence intervals
   - Compute effect size (Cohen's d)
   - Visualize results with boxplots and bar charts
   - Interpret business impact and recommendations

---

## Key Findings

- **p-value > 0.05:** No statistically significant difference in purchase rates between the bidding strategies.
- **Business Recommendation:** Continue with current bidding method unless further/longer tests suggest otherwise.

---

## Usage

1. **Install Requirements**

    ```bash
    pip install pandas scipy matplotlib openpyxl
    ```

2. **Prepare Dataset**

    Place the Excel file `ab_testing.xlsx` under the `Dataset/` directory. Structure:
    ```
    Dataset/
        ab_testing.xlsx
    ```

3. **Run the Script**

    ```bash
    python AB_TESTING.py
    ```

4. **Outputs**

    - Statistical test results in the console
    - Summary tables and visualizations (displayed as plots)

---

## File Structure

```
AB_TESTING.py      # Main analysis script
Dataset/
    ab_testing.xlsx # Excel file containing data (not included)
README.md          # This file
```

---

## Notes

- You can adapt this script for similar A/B test analyses by changing the dataset and metric names.

---

---

## Author

[Beyza Canakci](https://github.com/BeyzaCanakci)
