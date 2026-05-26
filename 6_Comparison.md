---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: epoxy-diffusion
    language: python
    name: python3
---

# Comparison Range for the Training Data

```python
from ml_tools.utilities import load_dataframe
from ml_tools.data_exploration import (filter_subset_continuous, 
                                       reconstruct_from_schema,
                                       plot_value_distributions_multi, 
                                       plot_numeric_overview_boxplot_macro, 
                                       summarize_dataframe)
from ml_tools.schema import FeatureSchema

from paths import PM
from helpers.constants import TARGET_RANGE, TARGET, TARGET_UNIT
```

```python
assert isinstance(TARGET_RANGE, (list, tuple)) and len(TARGET_RANGE) == 2, "TARGET_RANGE must be a list or tuple of length 2."
assert isinstance(TARGET, str) and TARGET, "TARGET must be a non-empty string."
assert isinstance(TARGET_UNIT, str) and TARGET_UNIT, "TARGET_UNIT must be a non-empty string."
```

## Load Data and Feature Schema

```python
df, _ = load_dataframe(PM.imputed_file)
```

```python
schema = FeatureSchema.from_json(PM.engineering)
```

## Reconstruct categorical features from the schema

```python
df_reconstructed = reconstruct_from_schema(df=df, schema=schema, targets=[TARGET])
```

## Filter on the chosen range of the target variable

```python
df_range = filter_subset_continuous(df=df_reconstructed, 
                                    range_filters={TARGET: TARGET_RANGE},
                                    drop_filter_cols=True)
```

```python
summarize_dataframe(df_range)
```

## Plot Distributions

```python
plot_numeric_overview_boxplot_macro(df=df_range,
                                    save_dir=PM.comparison,
                                    plot_title=f"Train Data Distribution - {TARGET} {TARGET_RANGE[0]} to {TARGET_RANGE[1]}",
                                    handle_zero_variance="constant",
                                    font_scaling=1.5)
```

## Load Generated Dataset

```python
#TODO: Chosen generated dataset for a specific target value
df_generated, _ = load_dataframe("results/Generation/Target-80-Guidance-3_0/Generated-500-samples.csv")
chosen_target_value = "80"
```

```python
assert len(chosen_target_value) > 0, "Set chosen_target_value to a valid value for the generated dataset."
```

```python
summarize_dataframe(df_generated)
```

```python
#TODO: Must have same column names and dtypes (cast int to float if necessary)
named_datasets = {f"Train Data ({TARGET_RANGE[0]}-{TARGET_RANGE[1]} {TARGET_UNIT})": df_range, 
                  f"Generated Data ({chosen_target_value} {TARGET_UNIT})": df_generated} 
```

## Plot Comparison Distributions

```python
plot_value_distributions_multi(named_dataframes=named_datasets,
                               save_dir=PM.comparison,
                               font_scaling=1.5,
                               mode="percentage")
```
