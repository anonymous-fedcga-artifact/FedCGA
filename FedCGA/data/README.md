# Data

Raw trip-level data and processed client feature files are not included in this anonymous artifact because of privacy and licensing restrictions.

Place the raw CSV files in this directory before running preprocessing:

```text
data/raw/train_set.csv
data/raw/val_set.csv
data/raw/test_set.csv
```

The expected CSV columns are:

```text
car_id, date, trip_start_time, trip_end_time, daytype, month, day,
slo, sla, elo, ela, traj, weather_0, weather_1, weather_2,
weather_3, weather_4, weather_5, dp_cur
```

After preprocessing, client-level feature files will be generated under:

```text
data/processed/train_features/
data/processed/val_features/
data/processed/test_features/
```
