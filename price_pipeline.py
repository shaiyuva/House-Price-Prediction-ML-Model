def train_price_model(data_path="data/train.csv"):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(data_path)
    df = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1)

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    model = GradientBoostingRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    return model