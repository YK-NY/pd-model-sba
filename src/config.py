"""Settings shared across the project, defined once."""

# Time-based split: training loans are ApprovalFY <= this year, test loans are later.
# Anything learned from outcomes (geography rates, WOE bins, IV table) uses training loans only.
TRAIN_YEARS_MAX = 2018
