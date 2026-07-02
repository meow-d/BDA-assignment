# Hourly ridership demand forecasting for the Komuter service

\>big data
\>looks inside
\>small data

## folder structure
- dataset - csv dataset (git lfs? never heard of her)
  - komuter_2026.csv - original from gov website
  - komuter_datetime.csv - date and time combined
- python - lightgbm version. uses uv
- rapidminer - deep learning version. rapidminer project and imported rapidminer dataset

## lightgbm version
```
# run
cd python
source .venv/bin/activate
uv run main.py # train model, evaluate, and visualize
uv run main.py predict # same as default but uses existing saved model
uv run main.py tune # same as default but runs hyperparameter tuning. takes hours.

# packages
uv add <package>

# check
uvx ty check
pyright main.py # alternative
```

## agent instructions
- no comments
- code must be as simple as possible
