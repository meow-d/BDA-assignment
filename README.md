# Hourly ridership demand forecasting for the Komuter service

## folder structure
- dataset - csv dataset (git lfs? never heard of her)
  - komuter_2026.csv - original from gov website
  - komuter_datetime.csv - date and time combined
- python - alternative python version just for testing. uses uv
- rapidminer - rapidminer project, and imported rapidminer dataset

## common commands
```
# run
cd python
source .venv/bin/activate.fish
uv run main.py # train
uv run main.py predict # use existing model

# packages
uv add <package>

# check
uvx ty check
pyright main.py # alternative
```

## agent instructions
- no comments
- code must be as simple as possible
