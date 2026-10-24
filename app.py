"""Penguin species prediction with K-nearest neighbors"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/"
    "palmerpenguins/main/inst/extdata/penguins.csv"
)
ISLANDS = ["Biscoe", "Dream", "Torgersen"]
SEXES = ["female", "male"]


@st.cache_resource
def train_model():
    """
    Load the data, select K by cross-validation, and fit the final pipeline

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Fitted StandardScaler and KNeighborsClassifier pipeline
    best_k : int
        Number of neighbors with the lowest cross-validation error
    X_train : pandas.DataFrame
        Training predictors
    y_train : pandas.Series
        Training species labels
    df_clean : pandas.DataFrame
        Penguin data after dropping rows with missing values
    """
    df = pd.read_csv(DATA_URL)
    df_clean = df.dropna()

    # Target is species, observation year is excluded
    y = df_clean["species"]
    X = df_clean.drop(columns=["species", "year"])

    # Island dummy columns with Biscoe as the reference category
    X = pd.get_dummies(X, columns=["island"], dtype=float, drop_first=True)

    # Female coded 0 and male coded 1
    X["sex_F_is_0"] = (X["sex"] == "male").astype(float)
    X = X.drop(columns="sex")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101, stratify=y
    )

    # Five-fold cross-validation error rate for K = 1 to 30, training data only
    k_values = list(range(1, 31))
    cv_errors = []
    for k in k_values:
        candidate = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores = cross_val_score(candidate, X_train, y_train, cv=5, scoring="accuracy")
        cv_errors.append(1 - scores.mean())

    # Smallest K with the lowest error
    best_k = k_values[cv_errors.index(min(cv_errors))]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ])
    model.fit(X_train, y_train)
    return model, best_k, X_train, y_train, df_clean


def make_penguin_row(columns, measurements, island, sex):
    """
    Build a one-row table of predictors in the training column order

    Parameters
    ----------
    columns : pandas.Index
        Predictor column names from the training data
    measurements : dict
        Values for bill length, bill depth, flipper length, and body mass
    island : str
        Island name, one of Biscoe, Dream, or Torgersen
    sex : str
        Either female or male

    Returns
    -------
    pandas.DataFrame
        One row with the same columns as the training predictors
    """
    row = dict(measurements)
    row["island_Dream"] = 1.0 if island == "Dream" else 0.0
    row["island_Torgersen"] = 1.0 if island == "Torgersen" else 0.0
    row["sex_F_is_0"] = 1.0 if sex == "male" else 0.0
    return pd.DataFrame([row]).reindex(columns=columns)


st.title("Penguin Species Predictor")
st.write(
    "This app predicts whether a penguin is an Adelie, Chinstrap, or Gentoo "
    "from its measurements, island, and sex. The model is a K-nearest-neighbors "
    "classifier trained on the Palmer Penguins data, with the number of "
    "neighbors chosen by five-fold cross-validation."
)

with st.spinner("Training the model"):
    model, best_k, X_train, y_train, df_clean = train_model()

st.subheader("Enter the penguin characteristics")

left, right = st.columns(2)
with left:
    bill_length = st.slider(
        "Bill length (mm)",
        float(df_clean["bill_length_mm"].min()),
        float(df_clean["bill_length_mm"].max()),
        float(df_clean["bill_length_mm"].median()),
        step=0.1,
    )
    bill_depth = st.slider(
        "Bill depth (mm)",
        float(df_clean["bill_depth_mm"].min()),
        float(df_clean["bill_depth_mm"].max()),
        float(df_clean["bill_depth_mm"].median()),
        step=0.1,
    )
    island = st.selectbox("Island", ISLANDS)
with right:
    flipper_length = st.slider(
        "Flipper length (mm)",
        float(df_clean["flipper_length_mm"].min()),
        float(df_clean["flipper_length_mm"].max()),
        float(df_clean["flipper_length_mm"].median()),
        step=1.0,
    )
    body_mass = st.slider(
        "Body mass (g)",
        float(df_clean["body_mass_g"].min()),
        float(df_clean["body_mass_g"].max()),
        float(df_clean["body_mass_g"].median()),
        step=25.0,
    )
    sex = st.selectbox("Sex", SEXES)

if st.button("Predict Species"):
    new_penguin = make_penguin_row(
        X_train.columns,
        {
            "bill_length_mm": bill_length,
            "bill_depth_mm": bill_depth,
            "flipper_length_mm": flipper_length,
            "body_mass_g": body_mass,
        },
        island,
        sex,
    )

    prediction = model.predict(new_penguin)[0]
    probabilities = model.predict_proba(new_penguin)[0]

    st.success(f"Predicted species: {prediction}")

    st.subheader("Predicted probabilities")
    probability_table = pd.DataFrame({
        "Species": model.classes_,
        "Probability": probabilities.round(2),
    })
    st.dataframe(probability_table, hide_index=True)

    st.subheader("Selected K")
    st.write(f"K = {best_k}, selected by five-fold cross-validation on the training data")

    # Nearest training penguins in the standardized predictor space
    scaled_penguin = model.named_steps["scaler"].transform(new_penguin)
    _, neighbor_positions = model.named_steps["knn"].kneighbors(scaled_penguin)
    neighbor_species = y_train.iloc[neighbor_positions[0]].value_counts()
    neighbor_text = ", ".join(
        f"{count} {species}" for species, count in neighbor_species.items()
    )

    st.subheader("Explanation")
    st.write(
        f"The model first standardizes each input so that no measurement "
        f"dominates the distance calculation. It then finds the {best_k} "
        f"training penguins closest to the entered penguin. Among those "
        f"{best_k} neighbors there are {neighbor_text}. The predicted "
        f"probability for each species is the share of the {best_k} neighbors "
        f"belonging to that species."
    )
