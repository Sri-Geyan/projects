base_df = spark.sql("""
SELECT *
FROM loan_applications l
JOIN credit_bureau b USING (borrower_id)
JOIN transactions t USING (borrower_id)
""")

base_df.display()
