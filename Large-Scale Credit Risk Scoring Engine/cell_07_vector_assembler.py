from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=['dti_ratio', 'credit_utilization', 'delinquency_flag', 'credit_score'],
    outputCol='features'
)

ml_df = assembler.transform(features_df).select('borrower_id', 'features', 'default')
