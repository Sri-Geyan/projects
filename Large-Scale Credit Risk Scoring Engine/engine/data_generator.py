import pandas as pd
import numpy as np

def generate_pandas_data(n=1000):
    """
    Generates synthetic synthetic data locally using pure pandas/numpy
    to recreate the pyspark random behavior roughly.
    """
    # Loan Applications
    loan_df = pd.DataFrame({
        'borrower_id': np.arange(n),
        'loan_amount': np.random.uniform(100000, 1000000, n).astype(int),
        'loan_tenure': np.random.uniform(12, 60, n).astype(int),
        'interest_rate': np.random.uniform(8, 18, n),
        'default': np.random.choice([0, 1], p=[0.8, 0.2], size=n)
    })
    
    # Credit Bureau
    bureau_df = pd.DataFrame({
        'borrower_id': np.arange(n),
        'credit_score': np.random.uniform(550, 850, n).astype(int),
        'total_credit': np.random.uniform(200000, 1000000, n).astype(int),
        'used_credit': np.random.uniform(0, 500000, n).astype(int),
        'delinquency_count': np.random.uniform(0, 5, n).astype(int)
    })
    
    # Transactions
    txn_df = pd.DataFrame({
        'borrower_id': np.arange(n),
        'monthly_income': np.random.uniform(15000, 135000, n).astype(int),
        'emi_outflow': np.random.uniform(0, 40000, n).astype(int)
    })
    
    return loan_df, bureau_df, txn_df

def generate_spark_data(spark, n=10000):
    """
    Generates the original synthetic spark databanks.
    """
    from pyspark.sql.functions import rand, when
    
    loan_df = (
        spark.range(n).withColumnRenamed('id', 'borrower_id')
        .withColumn('loan_amount', (rand()*900000 + 100000).cast('int'))
        .withColumn('loan_tenure', (rand()*48 + 12).cast('int'))
        .withColumn('interest_rate', rand()*10 + 8)
        .withColumn('default', when(rand() < 0.2, 1).otherwise(0))
    )
    
    bureau_df = (
        spark.range(n).withColumnRenamed('id', 'borrower_id')
        .withColumn('credit_score', (rand()*300 + 550).cast('int'))
        .withColumn('total_credit', (rand()*800000 + 200000).cast('int'))
        .withColumn('used_credit', (rand()*500000).cast('int'))
        .withColumn('delinquency_count', (rand()*5).cast('int'))
    )
    
    txn_df = (
        spark.range(n).withColumnRenamed('id', 'borrower_id')
        .withColumn('monthly_income', (rand()*120000 + 15000).cast('int'))
        .withColumn('emi_outflow', (rand()*40000).cast('int'))
    )
    
    return loan_df, bureau_df, txn_df
