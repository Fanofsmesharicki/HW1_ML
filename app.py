
import streamlit as st
import pandas as pd

st.title("Прогноз стоимости автомобилей")

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


@st.cache_resource
def load_model():
    with open("model.pickle", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts


@st.cache_data
def load_train():
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')

    df_train = df_train.drop_duplicates(subset=df_train.drop("selling_price",axis = 1), keep='first')
    df_train = df_train.reset_index(drop=True)

    df_train["mileage"] = df_train["mileage"].str[:-5].astype("float32")
    df_train["engine"] = df_train["engine"].str[:-3].astype("float32")

    df_train["max_power"] = df_train["max_power"].str[:-4]
    df_train = df_train[df_train["max_power"] != ""].copy()
    df_train["max_power"] = df_train["max_power"].astype("float32")

    df_train['torque'] = df_train['torque'].str.replace(r'\s+at\s+|@', '@', regex=True)
    ser_to_torque = df_train['torque'].str.split('@', expand=True)[0]
    ser_to_max_torq = df_train['torque'].str.split('@', expand=True)[1]

    df_train["torque"] = ser_to_torque
    df_train["max_torque_rpm"] = ser_to_max_torq

    mask_2= (df_train["torque"].str.contains("kgm", case = False , na = False) | df_train["max_torque_rpm"].str.contains("kgm", case = False , na = False))
    df_train["torque"] = df_train["torque"].str.replace(r"[A-Za-z]+", "", regex=True)

    mask  = (df_train.index == 633)
    df_train.loc[mask , "torque"] = "38.7"
    df_train.loc[df_train.index[1658], "torque"] = "400"

    mask_1 = (df_train["name"].str.contains("Skoda Octavia"))&(df_train["torque"].str.contains("/"))
    df_train.loc[mask_1 , "torque"] = "210"

    df_train["torque"] = pd.to_numeric(df_train["torque"])
    df_train.loc[mask_2, "torque"] *= 10

    df_train["max_torque_rpm"] = pd.to_numeric(df_train["max_torque_rpm"].str.replace(",", ".", regex=False).str.split("-").str[-1].str.replace(r"[^0-9]+", "", regex=True))
    medians = {}
    for column in df_train.select_dtypes("number").columns.to_list():
        fil = df_train[column].quantile(0.5)
        medians[column] = fil
        df_train[column] = df_train[column].fillna(fil)
    df_train[["engine","seats"]] = df_train[["engine","seats"]].astype("int8")

    list_cats = ["fuel", "seller_type", "transmission", "owner", "seats"]
    categories = {column: sorted(df_train[column].dropna().astype("str").unique())
        for column in list_cats}

    return df_train, medians, categories
def create_features(df):
    df["car_age"] = 2020 - df["year"]
    df["km_per_year"]=df["km_driven"]/(df["car_age"] + 1)
    df["log_km_driven"]=np.log1p(df["km_driven"].clip(lower=0))
    df["power_per_torque"]=(df["max_power"]/df["torque"].replace(0, np.nan))
    df["power_age"]=df["max_power"] * df["car_age"]
    df["mileage_age"]=df["mileage"] * df["car_age"]
    df["car_age_squared"]=df["car_age"]** 2
    df["log_max_power"]=np.log1p(df["max_power"].clip(lower=0))
    df["log_mileage"]=np.log1p(df["mileage"].clip(lower=0))
    df["power_squared"]=df["max_power"]**2
    df["power_mileage"]=df["max_power"] * df["mileage"]
    df["age_km_interaction"]=df["car_age"]*df["km_driven"]
    return df
def preprocess_test(df_test, medians, categories, encoder, feature_names):
    df_test = df_test.copy()
    df_test["mileage"] = df_test["mileage"].str[:-5].astype("float32")
    df_test["engine"] = df_test["engine"].str[:-3].astype("float32")
    df_test["max_power"] = df_test["max_power"].str[:-4]
    df_test = df_test[df_test["max_power"] != ""].copy()
    df_test["max_power"] = df_test["max_power"].astype("float32")
    df_test['torque'] = df_test['torque'].str.replace(r'\s+at\s+|@', '@', regex=True)
    ser_to_torque = df_test['torque'].str.split('@', expand=True)[0]
    ser_to_max_torq = df_test['torque'].str.split('@', expand=True).reindex(columns=[1])[1]
    df_test["torque"] = ser_to_torque
    df_test["max_torque_rpm"] = ser_to_max_torq
    mask_2= (df_test["torque"].str.contains("kgm", case = False , na = False) | df_test["max_torque_rpm"].str.contains("kgm", case = False , na = False))
    df_test["torque"] = df_test["torque"].str.replace(r"[A-Za-z]+", "", regex=True)
    mask_1 = (df_test["name"].str.contains("Skoda Octavia"))&(df_test["torque"].str.contains("/"))
    df_test.loc[mask_1 , "torque"] = "210"
    df_test["torque"] = pd.to_numeric(df_test["torque"], errors="coerce")
    df_test.loc[mask_2, "torque"] *= 10
    df_test["max_torque_rpm"] = pd.to_numeric(df_test["max_torque_rpm"].str.replace(",", ".", regex=False).str.split("-").str[-1].str.replace(r"[^0-9]+", "", regex=True))
    for column in medians:
        if column in df_test.columns:
            fil = medians[column]
            df_test[column] = df_test[column].fillna(fil)
    df_test[["engine","seats"]] = df_test[["engine","seats"]].astype("int8")
    X_test_cat = df_test.drop(columns = ["selling_price"], errors="ignore")
    X_test_cat["name"] = encoder.transform(X_test_cat[["name"]])
    X_test_cat["seats"] = X_test_cat["seats"].astype("str")
    list_cats = ["fuel", "seller_type", "transmission", "owner", "seats"]
    for column in list_cats:
        X_test_cat[column] = pd.Categorical(
            X_test_cat[column].astype("str"),
            categories=categories[column])

    X_test_cat = pd.get_dummies(X_test_cat, drop_first = True)
    X_test_cat = create_features(X_test_cat)
    X_test_cat = X_test_cat.reindex(columns=feature_names,fill_value=0)
    return X_test_cat
artifacts = load_model()
model = artifacts["model"]
encoder = artifacts["encoder"]
feature_names = artifacts["feature_names"]
df_train, medians, categories = load_train()
st.title("Предсказание стоимости автомобилей")
st.header("EDA")
st.subheader("Распределение стоимости автомобилей")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df_train, x='selling_price', bins=40, ax=ax)
st.pyplot(fig)
plt.close(fig)
st.subheader("Корреляция признаков")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_train.select_dtypes(["number"]).corr(),annot = True, ax=ax)
st.pyplot(fig)
plt.close(fig)
st.subheader("Стоимость автомобилей по категориальным признакам")
fig, axes =plt.subplots(1,3,figsize=(18, 5))
sns.boxplot(data=df_train, x='fuel', y='selling_price', ax=axes[0])
sns.boxplot(data=df_train, x='transmission', y='selling_price', ax=axes[1])
sns.boxplot(data=df_train, x='owner', y='selling_price', ax=axes[2])
st.pyplot(fig)
plt.close(fig)
st.header("Предсказание модели")
uploaded_file = st.file_uploader("Загрузите CSV-файл с признаками автомобилей", type=["csv"])
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    required_columns = [
        "name", "year", "km_driven", "fuel", "seller_type","transmission", "owner", "mileage", "engine","max_power", "torque", "seats"]
    missing_columns = [column for column in required_columns if column not in df_test.columns]
    if missing_columns:
        st.error(f"Отсутствуют столбцы: {missing_columns}")
    elif df_test.empty:
        st.warning("CSV-файл пустой")
    else:
        try:
            X_test_cat = preprocess_test(df_test, medians, categories, encoder, feature_names)
            if X_test_cat.empty:
                st.warning("Нет объектов для предсказания")
            else:
                predictions = model.predict(X_test_cat)
                result = df_test.loc[X_test_cat.index].copy()
                result["predicted_selling_price"] = predictions
                st.subheader("Результаты предсказания")
                st.dataframe(result)
        except (ValueError, TypeError, KeyError) as e:
            st.error(f"Ошибка предобработки: {e}")
st.header("Коэффициенты модели Ridge")
coefficients = pd.Series(model.coef_, index=feature_names)
coefficients = coefficients.sort_values()
fig, ax = plt.subplots(figsize=(10, max(6, len(coefficients) * 0.3)))
coefficients.plot(kind="barh", ax=ax)
ax.set_xlabel("Значение коэффициента")
ax.set_ylabel("Признак")
ax.set_title("Коэффициенты обученной модели")
ax.axvline(0, color="black")

st.pyplot(fig)
plt.close(fig)
