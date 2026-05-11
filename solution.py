import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "meteo"


def load_raw_data():
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    query = "SELECT * FROM observations"
    df = pd.read_sql(query, engine)
    return df


def block_1_numpy(raw_df):
    print("\n========== БЛОК 1. NUMPY ==========")

    obs_ids = raw_df["obs_id"].to_numpy()
    datetimes = raw_df["datetime"].to_numpy()

    temperature = raw_df["temperature_c"].to_numpy(dtype=float)
    humidity = raw_df["humidity_pct"].to_numpy(dtype=float)
    wind_speed = raw_df["wind_speed_ms"].to_numpy(dtype=float)

    apparent_temperature = temperature - (100 - humidity) / 5

    temperature_clean_np = np.where(
        (temperature > 60) | (temperature < -60),
        np.nan,
        temperature
    )

    wind_speed_clean_np = np.where(
        wind_speed > 100,
        np.nan,
        wind_speed
    )

    valid_temp_mask = ~np.isnan(temperature_clean_np)
    n = np.sum(valid_temp_mask)

    temp_mean = np.nansum(temperature_clean_np) / n
    temp_median = np.nanmedian(temperature_clean_np)
    temp_std = np.sqrt(
        np.nansum((temperature_clean_np[valid_temp_mask] - temp_mean) ** 2) / n
    )

    frost_count = np.sum(temperature_clean_np < 0)
    hot_count = np.sum(temperature_clean_np > 30)

    max_idx = np.nanargmax(temperature_clean_np)
    min_idx = np.nanargmin(temperature_clean_np)

    print(f"Apparent temperature, перші 10 значень: {apparent_temperature[:10]}")
    print(f"Середня температура: {temp_mean:.2f}")
    print(f"Медіана температури: {temp_median:.2f}")
    print(f"Стандартне відхилення температури: {temp_std:.2f}")
    print(f"Кількість морозних спостережень T < 0: {frost_count}")
    print(f"Кількість жарких спостережень T > 30: {hot_count}")

    print("\nМаксимальна температура:")
    print(f"obs_id: {obs_ids[max_idx]}")
    print(f"datetime: {datetimes[max_idx]}")
    print(f"temperature_c: {temperature_clean_np[max_idx]:.2f}")

    print("\nМінімальна температура:")
    print(f"obs_id: {obs_ids[min_idx]}")
    print(f"datetime: {datetimes[min_idx]}")
    print(f"temperature_c: {temperature_clean_np[min_idx]:.2f}")

    return temperature_clean_np, wind_speed_clean_np


def block_2_cleaning(raw_df):
    print("\n========== БЛОК 2. PANDAS — ОЧИЩЕННЯ ==========")

    df = raw_df.copy()

    rows_before = len(df)

    print("\nІнформація про таблицю:")
    print(df.info())

    print("\nОпис числових колонок:")
    print(df.describe())

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    rows_before_duplicates = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before_duplicates - len(df)

    humidity_na_before = df["humidity_pct"].isna().sum()

    df["month"] = df.index.month
    df["humidity_pct"] = df.groupby(["city", "month"])["humidity_pct"].transform(
        lambda s: s.fillna(s.median())
    )

    humidity_na_after = df["humidity_pct"].isna().sum()
    humidity_filled = humidity_na_before - humidity_na_after

    rows_before_outliers = len(df)

    df = df[
        (df["temperature_c"] >= -60) &
        (df["temperature_c"] <= 60) &
        (
            df["wind_speed_ms"].isna() |
            (
                (df["wind_speed_ms"] >= 0) &
                (df["wind_speed_ms"] <= 60)
            )
        )
    ]

    outliers_removed = rows_before_outliers - len(df)
    rows_after = len(df)

    print("\nЗвіт очищення:")
    print(f"Рядків було: {rows_before}")
    print(f"Рядків стало: {rows_after}")
    print(f"Видалено дублів: {duplicates_removed}")
    print(f"Заповнено NaN у humidity_pct: {humidity_filled}")
    print(f"Видалено рядків з фізичними викидами: {outliers_removed}")

    return df


def block_3_analytics(df):
    print("\n========== БЛОК 3. PANDAS — АНАЛІТИКА ==========")

    mean_temp_by_city = df.groupby("city")["temperature_c"].mean().sort_values()
    print("\nСередня температура по містах:")
    print(mean_temp_by_city)

    coldest_city = mean_temp_by_city.idxmin()
    warmest_city = mean_temp_by_city.idxmax()

    print(f"\nНайхолодніше місто: {coldest_city}")
    print(f"Найтепліше місто: {warmest_city}")

    precipitation_by_city = df.groupby("city")["precipitation_mm"].sum().sort_values(
        ascending=False
    )
    print("\nСумарні опади по містах:")
    print(precipitation_by_city)

    wettest_city = precipitation_by_city.idxmax()
    print(f"\nНайвологіше місто: {wettest_city}")

    try:
        monthly_temp = df.groupby("city")["temperature_c"].resample("ME").mean()
    except ValueError:
        monthly_temp = df.groupby("city")["temperature_c"].resample("M").mean()

    print("\nМісячна середня температура:")
    print(monthly_temp.head(20))

    pivot_city_month = pd.pivot_table(
        df,
        values="temperature_c",
        index="city",
        columns="month",
        aggfunc="mean"
    )

    print("\nPivot table: місто × місяць, середня температура:")
    print(pivot_city_month)

    daily_precipitation = (
        df.groupby("city")
        .resample("D")["precipitation_mm"]
        .sum()
        .reset_index()
    )

    rainy_days = (
        daily_precipitation[daily_precipitation["precipitation_mm"] > 5]
        .groupby("city")
        .size()
    )

    print("\nКількість днів з опадами > 5 мм по містах:")
    print(rainy_days)

    df_anomaly = df.copy()
    df_anomaly["year"] = df_anomaly.index.year
    df_anomaly["month"] = df_anomaly.index.month

    monthly_by_year = (
        df_anomaly
        .groupby(["year", "month"])["temperature_c"]
        .mean()
        .reset_index()
    )

    monthly_norm = (
        monthly_by_year
        .groupby("month")["temperature_c"]
        .mean()
        .rename("norm_temperature")
        .reset_index()
    )

    anomaly_table = monthly_by_year.merge(monthly_norm, on="month")
    anomaly_table["deviation"] = (
        anomaly_table["temperature_c"] - anomaly_table["norm_temperature"]
    )
    anomaly_table["abs_deviation"] = anomaly_table["deviation"].abs()

    anomaly_row = anomaly_table.loc[anomaly_table["abs_deviation"].idxmax()]

    anomaly_year = int(anomaly_row["year"])
    anomaly_month = int(anomaly_row["month"])
    anomaly_deviation = anomaly_row["deviation"]

    if anomaly_deviation > 0:
        anomaly_type = "хвиля спеки"
    else:
        anomaly_type = "холодна хвиля"

    print("\nАномальний місяць:")
    print(f"Рік: {anomaly_year}")
    print(f"Місяць: {anomaly_month}")
    print(f"Відхилення: {anomaly_deviation:.2f} °C")
    print(f"Тип аномалії: {anomaly_type}")

    results = {
        "mean_temp_by_city": mean_temp_by_city,
        "coldest_city": coldest_city,
        "warmest_city": warmest_city,
        "precipitation_by_city": precipitation_by_city,
        "wettest_city": wettest_city,
        "monthly_temp": monthly_temp,
        "pivot_city_month": pivot_city_month,
        "rainy_days": rainy_days,
        "anomaly_year": anomaly_year,
        "anomaly_month": anomaly_month,
        "anomaly_deviation": anomaly_deviation,
        "anomaly_type": anomaly_type,
    }

    return results


def block_4_plots(df, results):
    print("\n========== БЛОК 4. MATPLOTLIB ==========")

    os.makedirs("plots", exist_ok=True)

    cities = df["city"].dropna().unique()[:3]

    try:
        monthly_city_temp = (
            df[df["city"].isin(cities)]
            .groupby("city")["temperature_c"]
            .resample("ME")
            .mean()
            .reset_index()
        )
    except ValueError:
        monthly_city_temp = (
            df[df["city"].isin(cities)]
            .groupby("city")["temperature_c"]
            .resample("M")
            .mean()
            .reset_index()
        )

    plt.figure(figsize=(10, 6))

    for city in cities:
        city_data = monthly_city_temp[monthly_city_temp["city"] == city]
        plt.plot(
            city_data["datetime"],
            city_data["temperature_c"],
            marker="o",
            label=city
        )

    plt.title("Місячна динаміка температури для трьох міст")
    plt.xlabel("Дата")
    plt.ylabel("Середня температура, °C")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/01_monthly_temperature_line.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    results["precipitation_by_city"].plot(kind="bar")
    plt.title("Сумарні опади по містах")
    plt.xlabel("Місто")
    plt.ylabel("Опади, мм")
    plt.tight_layout()
    plt.savefig("plots/02_precipitation_bar.png", dpi=150)
    plt.close()

    temp_mean = df["temperature_c"].mean()
    temp_median = df["temperature_c"].median()

    plt.figure(figsize=(10, 6))
    plt.hist(df["temperature_c"].dropna(), bins=30, edgecolor="black")
    plt.axvline(temp_mean, linestyle="--", label=f"Mean = {temp_mean:.2f}")
    plt.axvline(temp_median, linestyle="-", label=f"Median = {temp_median:.2f}")
    plt.title("Розподіл температури")
    plt.xlabel("Температура, °C")
    plt.ylabel("Кількість спостережень")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/03_temperature_histogram.png", dpi=150)
    plt.close()

    pivot = results["pivot_city_month"]

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Середня температура, °C")
    plt.title("Heatmap: середня температура за містами і місяцями")
    plt.xlabel("Місяць")
    plt.ylabel("Місто")
    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=pivot.columns
    )
    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=pivot.index
    )
    plt.tight_layout()
    plt.savefig("plots/04_temperature_heatmap.png", dpi=150)
    plt.close()

    print("Графіки збережено в папку plots/")


def main():
    raw_df = load_raw_data()

    block_1_numpy(raw_df)

    clean_df = block_2_cleaning(raw_df)

    results = block_3_analytics(clean_df)

    block_4_plots(clean_df, results)


if __name__ == "__main__":
    main()


"""
Висновки:
Після очищення даних були видалені повні дублікати та фізично неможливі
викиди температури і швидкості вітру, тому подальші розрахунки виконані
на більш надійному наборі даних. Найтепліше місто та найхолодніше місто
визначаються за середньою температурою по містах, яка виводиться у Блоці 3.
Сезонність у даних виражена чітко: на лінійному графіку середньомісячна
температура зростає в теплий період року та знижується в холодний період.
Найвологіше місто визначається за найбільшою сумою опадів за весь період
спостережень, що також розраховується у Блоці 3 і показується на bar-графіку.
Аномальний місяць знайдено як пару рік-місяць із найбільшим абсолютним
відхиленням від кліматичної норми для відповідного календарного місяця.
Додатне відхилення означає хвилю спеки, а від’ємне — холодну хвилю.
Для оцінки стабільності клімату варто порівнювати міста або регіони за
розмахом середньомісячних температур: менший розмах означає стабільніші
температурні умови. Рекомендовано звернути увагу на міста з найбільшою
кількістю днів з опадами понад 5 мм, оскільки вони можуть мати вищі ризики
підтоплень або потребувати кращого планування водовідведення.
"""
