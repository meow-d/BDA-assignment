# Hourly ridership demand forecasting for the Komuter service

\>big data

\>looks inside

\>small data

## folder structure
- dataset - csv dataset (git lfs required)
  - komuter_2026.csv - original from gov website
  - komuter_datetime.csv - date and time combined
- lightgbm - lightgbm version. uv project
- autots - autots version. uv project
- rapidminer - processes for various rapidminer versions, plus imported rapidminer dataset

## lightgbm version
```
# run
cd lightgbm
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

## autots version
```
cd autots
source
uv run main.py # run the automl, saves model template
uv run main.py --predict # predict using already saved model template
```

## agent instructions
- no comments
- code must be as simple as possible
