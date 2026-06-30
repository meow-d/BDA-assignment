# Hourly ridership demand forecasting for the Komuter service

## folder structure
- dataset - csv dataset (git lfs? never heard of her)
  - komuter_2026.csv - original from gov website
  - komuter_datetime.csv - date and time combined
- python - alternative python version just for testing. uses uv
- rapidminer - rapidminer project, and imported rapidminer dataset

## commands
```
cd python
uv run main.py # train
uv run main.py predict # use existing model
```

## agent instructions
- no comments
- code must be as simple as possible
