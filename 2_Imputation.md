---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: dragon-mice
    language: python
    name: python3
---

```python
from ml_tools.MICE import DragonMICE
from ml_tools.VIF import compute_vif_multi
from ml_tools.utilities import load_dataframe_greedy, save_dataframe_with_schema
from ml_tools.schema import FeatureSchema

from paths import PM
from helpers.constants import TARGET
```

## 0. Load Feature Schema

```python
schema = FeatureSchema.from_json(PM.engineering)
```

## 1. Imputation

```python
imputer = DragonMICE(schema=schema,
                     impute_targets=False,
                     iterations=30,
                     resulting_datasets=1)
```

```python
imputer.run_pipeline(df_path_or_dir=PM.engineering_file,
                     save_datasets_dir=PM.mice_datasets,
                     save_metrics_dir=PM.mice)
```

## 2. VIF Analysis

```python
compute_vif_multi(input_directory=PM.mice_datasets,
                  output_plot_directory=PM.vif,
                  ignore_columns=[TARGET])
```

## 3. Save Imputed Dataset

```python
df_imputed = load_dataframe_greedy(directory=PM.mice_datasets)
```

```python
save_dataframe_with_schema(df=df_imputed, full_path=PM.imputed_file, schema=schema)
```
