def predict(pipeline, X_test):
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    return probs, preds
