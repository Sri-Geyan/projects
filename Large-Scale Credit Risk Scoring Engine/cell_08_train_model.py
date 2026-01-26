from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol='features', labelCol='default', maxIter=50)
model = lr.fit(ml_df)
predictions = model.transform(ml_df)
