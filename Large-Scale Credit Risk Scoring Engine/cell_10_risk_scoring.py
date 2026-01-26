from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, when

predictions_clean = predictions.select('borrower_id', 'probability')

scored_df = predictions_clean.withColumn('risk_score', vector_to_array(col('probability'))[1])

scored_df = scored_df.withColumn(
    'risk_bucket',
    when(col('risk_score') > 0.7, 'High')
    .when(col('risk_score') > 0.4, 'Medium')
    .otherwise('Low')
)
