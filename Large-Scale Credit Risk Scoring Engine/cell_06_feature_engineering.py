from pyspark.sql.functions import col, when

features_df = (
    base_df
    .withColumn('dti_ratio', col('emi_outflow')*12/col('monthly_income'))
    .withColumn('credit_utilization', col('used_credit')/col('total_credit'))
    .withColumn('delinquency_flag', when(col('delinquency_count') > 0, 1).otherwise(0))
)

features_df.display()
