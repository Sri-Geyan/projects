from pyspark.sql.functions import rand, when

N = 1_000_000

loan_df = (
    spark.range(N).withColumnRenamed('id', 'borrower_id')
    .withColumn('loan_amount', (rand()*900000 + 100000).cast('int'))
    .withColumn('loan_tenure', (rand()*48 + 12).cast('int'))
    .withColumn('interest_rate', rand()*10 + 8)
    .withColumn('default', when(rand() < 0.2, 1).otherwise(0))
)

bureau_df = (
    spark.range(N).withColumnRenamed('id', 'borrower_id')
    .withColumn('credit_score', (rand()*300 + 550).cast('int'))
    .withColumn('total_credit', (rand()*800000 + 200000).cast('int'))
    .withColumn('used_credit', (rand()*500000).cast('int'))
    .withColumn('delinquency_count', (rand()*5).cast('int'))
)

txn_df = (
    spark.range(N).withColumnRenamed('id', 'borrower_id')
    .withColumn('monthly_income', (rand()*120000 + 15000).cast('int'))
    .withColumn('emi_outflow', (rand()*40000).cast('int'))
)
