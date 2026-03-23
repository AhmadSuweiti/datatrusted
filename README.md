# datatrust

**A trust layer for tabular data.**

`datatrust` helps analysts, data scientists, and ML engineers answer one question before training a model or running an analysis:

> **Can I trust this dataset?**

It audits a pandas DataFrame in one function call and returns a structured report with a trust score, plain-language warnings, and per-check details — then lets you export the whole thing to Markdown or HTML.

---

## Why datatrust?

Most data quality bugs surface late — after a model is in production, or after an analysis has been shared. The root causes are almost always things that could have been caught early:

- A column that looks numeric but is stored as strings
- A join key with silent duplicates that inflated row counts
- A target column with 40 % missing labels, unnoticed
- A "feature" that's actually the outcome encoded post-event
- A test set drawn from a different distribution than train

`datatrust` makes these checks fast, consistent, and automated. It is not a dashboard, a data catalog, or a monitoring system. It is a library you call before you do anything else with a dataset.

---

## Features

| Check | What it detects |
|---|---|
| **Schema / types** | Numeric data stored as strings, unparsed datetime columns, constant columns, high-cardinality object columns |
| **Missing values** | Per-column counts and percentages, configurable threshold flagging |
| **Duplicates** | Full-row duplicates, duplicate values in ID / key columns |
| **Outliers** | IQR-based outlier counts per numeric column (no auto-removal) |
| **Target analysis** | Missing labels, class imbalance, cardinality warnings |
| **Leakage hints** | Suspicious column names, near-perfect target correlation, near-unique key columns |
| **Train/test drift** | Numeric mean shift, categorical TVD, missing / unseen categories |
| **Join integrity** | Duplicate join keys, unmatched keys, many-to-many detection |
| **Rule validation** | Not-null, unique, numeric range, non-negative, date not in future, allowed values |
| **Trust score** | A single 0–100 score derived from all checks, with transparent penalties |

---

## Installation

```bash
pip install datatrust
```

Or install from source:

```bash
git clone https://github.com/yourusername/datatrust
cd datatrust
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, pandas ≥ 1.5, numpy ≥ 1.23. No other hard dependencies.

---

## Quickstart

### Full dataset audit

```python
import pandas as pd
from datatrust import audit

df = pd.read_csv("customers.csv")

report = audit(
    df,
    target="churn",
    id_columns=["customer_id"],
    missing_threshold=0.05,
)

print(report.score)    # e.g. 71
print(report.summary)  # plain-text paragraph

# Export
report.to_html("audit_report.html")
report.to_markdown("audit_report.md")
```

### Custom validation rules

```python
from datatrust import Validator, audit

validator = (
    Validator()
    .not_null("email")
    .unique("customer_id")
    .in_range("age", 0, 120)
    .non_negative("revenue")
    .date_not_in_future("signup_date")
    .allowed_values("status", ["active", "inactive", "pending"])
)

report = audit(df, validator=validator)

if not report.validation_result.is_valid:
    for v in report.validation_result.violations:
        print(f"[{v.rule}] {v.column}: {v.description}")
```

### Train / test drift

```python
from datatrust import compare_splits

drift = compare_splits(train_df, test_df, target="label")

print(f"Drifted columns: {drift.drifted_columns}")

for d in drift.numeric_drifts:
    if d.drift_detected:
        print(f"  {d.column}: {d.description}")

for d in drift.categorical_drifts:
    if d.missing_in_test:
        print(f"  {d.column}: categories absent in test — {d.missing_in_test}")
```

### Join integrity

```python
from datatrust import check_join

report = check_join(orders_df, customers_df, on="customer_id")

for warning in report.warnings:
    print(warning)

# e.g.:
# Left DataFrame has 142 row(s) with duplicate join key values on ['customer_id'].
# 38 row(s) in the left DataFrame have no matching key in the right DataFrame.
```

---

## Example output

```
Trust Score: 74/100

Dataset: 12,430 rows × 18 columns. Trust score: 74/100.
Missing values: 3 column(s) exceed the 5.0% threshold.
Duplicates: 47 full-row duplicate(s) (0.4%).
Schema: 2 issue(s) found (e.g. numeric-as-string, unparsed datetimes).
Target 'churn': 3.1% missing labels. Class imbalance ratio 8.2x.
[Leakage hint] Column 'outcome' has a name that may indicate it encodes the outcome.
```

**Warnings (excerpt):**
- Column `'contract_end_date'` is `12.3%` missing.
- Column `'revenue_str'` has dtype object but its values look numeric.
- Column `'outcome'` has a name that may indicate it encodes the outcome.
- Column `'churn'` has `385` null value(s) (`3.1%` missing labels).
- Column `'age'` has `214` outlier(s) (`1.7%`) outside `[-22.3, 103.7]`.

---

## API reference

### `audit(df, *, target, id_columns, datetime_columns, missing_threshold, validator, check_leakage) → AuditReport`

Runs all checks and returns an `AuditReport`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `DataFrame` | — | Dataset to audit |
| `target` | `str` | `None` | Target/label column name |
| `id_columns` | `list[str]` | `None` | Columns that should be unique identifiers |
| `datetime_columns` | `list[str]` | `None` | Columns the caller knows are datetimes |
| `missing_threshold` | `float` | `0.05` | Flag columns above this missing fraction |
| `validator` | `Validator` | `None` | Pre-configured validation rules to run |
| `check_leakage` | `bool` | `True` | Whether to run heuristic leakage checks |

### `AuditReport`

| Attribute / Method | Description |
|---|---|
| `.score` | Trust score 0–100 |
| `.summary` | Plain-text one-paragraph summary |
| `.warnings` | `list[str]` of human-readable issues |
| `.to_dict()` | JSON-serialisable dict |
| `.to_markdown(path=None)` | Markdown string, optionally written to file |
| `.to_html(path=None)` | Self-contained HTML string, optionally written to file |
| `.shape` | `(rows, cols)` of the audited DataFrame |
| `.schema_report` | `SchemaReport` with dtype map and issues |
| `.missing_info` | `MissingInfo` with per-column missing stats |
| `.duplicate_info` | `DuplicateInfo` |
| `.outlier_infos` | `list[OutlierInfo]` |
| `.target_info` | `TargetInfo` or `None` |
| `.leakage_hints` | `list[LeakageHint]` |
| `.validation_result` | `ValidationResult` or `None` |

### `compare_splits(train_df, test_df, target=None) → DriftReport`

### `check_join(left, right, on) → JoinReport`

### `Validator`

```python
Validator()
    .not_null(column)
    .unique(column)
    .in_range(column, min_val=None, max_val=None)
    .non_negative(column)
    .date_not_in_future(column)
    .allowed_values(column, allowed)
    .add_rule(rule)   # for custom Rule subclasses
    .validate(df) → ValidationResult
```

---

## Trust score

The score starts at 100 and deducts points for detected issues. The table below shows the penalty logic, which lives in `datatrust/report.py` and is easy to adjust:

| Issue | Penalty |
|---|---|
| Column 5–20 % missing | −2 per column |
| Column 20–50 % missing | −5 per column |
| Column > 50 % missing | −10 per column |
| Duplicates 0–1 % | −2 |
| Duplicates 1–5 % | −5 |
| Duplicates > 5 % | −10 |
| Each rule violation | −3 (capped at −20) |
| Each schema issue | −2 (capped at −10) |
| Target missing 2–10 % | −5 |
| Target missing > 10 % | −10 |
| Target imbalance ratio > 5x | −5 |
| Each outlier column | −1 (capped at −5) |
| Each high-severity leakage hint | −5 (capped at −15) |

The minimum score is 0.

---

## Project structure

```
datatrust/
├── datatrust/
│   ├── __init__.py       # Public API
│   ├── audit.py          # Main audit() entrypoint
│   ├── schema.py         # Schema / type checks
│   ├── missing.py        # Missing value analysis
│   ├── duplicates.py     # Duplicate detection
│   ├── rules.py          # Validator + Rule classes
│   ├── outliers.py       # IQR outlier detection
│   ├── target.py         # Target column analysis
│   ├── leakage.py        # Heuristic leakage hints
│   ├── drift.py          # Train/test drift comparison
│   ├── joins.py          # Join integrity checks
│   ├── report.py         # AuditReport + scoring + HTML/MD export
│   ├── models.py         # All dataclasses
│   └── utils.py          # Internal helpers
├── tests/
│   ├── conftest.py
│   ├── test_missing.py
│   ├── test_duplicates.py
│   ├── test_rules.py
│   ├── test_target.py
│   ├── test_drift.py
│   ├── test_joins.py
│   ├── test_report.py
│   └── test_audit.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Running tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=datatrust --cov-report=term-missing

# Run a specific test file
pytest tests/test_rules.py -v
```

---

## Contributing

Contributions are welcome. A few guidelines:

1. Keep new checks modular — one file per concern.
2. Return structured dataclasses, not raw dicts or strings.
3. Write tests for every new check.
4. Keep the public API small and stable.

To add a custom rule, subclass `datatrust.rules.Rule`:

```python
from datatrust.rules import Rule, Validator
from datatrust.models import RuleViolation
from typing import Optional
import pandas as pd

class NoWhitespaceRule(Rule):
    def __init__(self, column: str):
        self.column = column

    @property
    def name(self) -> str:
        return "no_whitespace"

    def validate(self, df: pd.DataFrame) -> Optional[RuleViolation]:
        series = df[self.column].dropna().astype(str)
        mask = series.str.contains(r"\s")
        count = int(mask.sum())
        if count == 0:
            return None
        return RuleViolation(
            rule=self.name,
            column=self.column,
            description=f"Column '{self.column}' has {count} value(s) containing whitespace.",
            affected_rows=count,
        )

result = Validator().add_rule(NoWhitespaceRule("username")).validate(df)
```

---

---

## License

MIT. See [LICENSE](LICENSE).
