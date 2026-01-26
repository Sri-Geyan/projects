from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol='default', metricName='areaUnderROC')
auc = evaluator.evaluate(predictions)
print(f'AUC-ROC: {auc}')
