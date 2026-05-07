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
from ml_tools.path_manager import make_fullpath

from paths import PM
from helpers.constants import TARGET_RANGE, TARGET
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
df_original = reconstruct_from_schema(df=df, schema=schema, targets=[TARGET])
```

## Filter on the chosen range of the target variable

```python
df_range = filter_subset_continuous(df=df_original, 
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
                                    plot_title=f"Train Data Distribution - {TARGET} range {TARGET_RANGE[0]} to {TARGET_RANGE[1]}",
                                    handle_zero_variance="constant")
```

## Plot Comparison

```python
generated_local_path = "results/Generation/Target-80-Guidance-3_0/Generated-500-samples.csv"

df_generated, _ = load_dataframe(df_path=generated_local_path)
```

```python
# Named Dataframes
named_dataframes = {"Train": df_range, "Generated": df_generated}
```

```python
plot_value_distributions_multi(named_dataframes=named_dataframes,
                               save_dir=PM.comparison,
                               font_scaling=1.5,
                               mode="percentage")
```
