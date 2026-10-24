# Penguin Species Classification

This project predicts the species of a penguin (Adelie, Chinstrap, or Gentoo) from bill length, bill depth, flipper length, body mass, island, and sex using K-nearest neighbors. A Streamlit app lets a user enter the characteristics of a new penguin and see the predicted species and class probabilities.

## Data

Palmer Penguins data, loaded directly from the public CSV file. Rows with missing values are dropped, leaving 333 of 344 penguins. Observation year is not used as a predictor.

## Model

- Island is encoded as dummy columns with Biscoe as the reference category, and sex is encoded as `sex_F_is_0` (female = 0, male = 1)
- Stratified split with 70 percent training and 30 percent test data (`random_state=101`)
- Pipeline with `StandardScaler` and `KNeighborsClassifier`
- K chosen from 1 to 30 by five-fold cross-validation on the training data, which selected K = 2
- Test accuracy of 0.99 on 100 test penguins

## Files

- `app.py`: Streamlit application
- `requirements.txt`: Python packages needed to run the app
- `analysis.ipynb`: Jupyter notebook with the data description, preparation, tuning, and evaluation

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Live app

Add the public Streamlit link here after deployment.
