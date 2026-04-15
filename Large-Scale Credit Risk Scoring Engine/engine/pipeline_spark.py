from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, when

class SparkRiskScoringPipeline:
    def __init__(self, spark):
        self.spark = spark
        
    def join_data(self, loan_df, bureau_df, txn_df):
        loan_df.createOrReplaceTempView('loan_applications')
        bureau_df.createOrReplaceTempView('credit_bureau')
        txn_df.createOrReplaceTempView('transactions')
        
        base_df = self.spark.sql("""
        SELECT *
        FROM loan_applications l
        JOIN credit_bureau b USING (borrower_id)
        JOIN transactions t USING (borrower_id)
        """)
        return base_df
        
    def feature_engineering(self, base_df):
        features_df = (
            base_df
            .withColumn('dti_ratio', col('emi_outflow')*12/col('monthly_income'))
            .withColumn('credit_utilization', col('used_credit')/col('total_credit'))
            .withColumn('delinquency_flag', when(col('delinquency_count') > 0, 1).otherwise(0))
        )
        return features_df
        
    def train_model(self, features_df):
        assembler = VectorAssembler(
            inputCols=['dti_ratio', 'credit_utilization', 'delinquency_flag', 'credit_score'],
            outputCol='features',
            handleInvalid="skip"
        )
        ml_df = assembler.transform(features_df).select('borrower_id', 'features', 'default')
        
        lr = LogisticRegression(featuresCol='features', labelCol='default', maxIter=50)
        model = lr.fit(ml_df)
        predictions = model.transform(ml_df)
        return predictions, model

    def evaluate_model(self, predictions):
        evaluator = BinaryClassificationEvaluator(labelCol='default', metricName='areaUnderROC')
        auc = evaluator.evaluate(predictions)
        return auc

    def calculate_risk_scores(self, predictions):
        predictions_clean = predictions.select('borrower_id', 'probability')
        scored_df = predictions_clean.withColumn('risk_score', vector_to_array(col('probability'))[1])
        
        scored_df = scored_df.withColumn(
            'risk_bucket',
            when(col('risk_score') > 0.7, 'High')
            .when(col('risk_score') > 0.4, 'Medium')
            .otherwise('Low')
        )
        return scored_df
