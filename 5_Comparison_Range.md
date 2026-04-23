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
                                       plot_value_distributions, 
                                       plot_numeric_overview_boxplot_macro, 
                                       summarize_dataframe)
from ml_tools.schema import FeatureSchema
from ml_tools.serde import serialize_object

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
plot_value_distributions(df=df_range, save_dir=PM.train_comparison)
```

```python
plot_numeric_overview_boxplot_macro(df=df_range,
                                    save_dir=PM.train_comparison,
                                    plot_title=f"Data Distribution - {TARGET} range {TARGET_RANGE[0]} to {TARGET_RANGE[1]}",
                                    handle_zero_variance="constant")
```

## Save batch size to file for use in generation script

```python
serialize_object(obj=df_range.shape[0], file_path=PM.batch_size_file)
```
