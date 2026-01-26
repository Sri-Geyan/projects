final_df = scored_df.select('borrower_id', 'risk_score', 'risk_bucket')
final_df.display()
