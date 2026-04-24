import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class PandasRiskScoringPipeline:
    def join_data(self, loan_df, bureau_df, txn_df):
        # Join dataframes on borrower_id
        base_df = loan_df.merge(bureau_df, on='borrower_id')
        base_df = base_df.merge(txn_df, on='borrower_id')
        return base_df
        
    def feature_engineering(self, base_df):
        features_df = base_df.copy()
        
        # dti_ratio = emi_outflow * 12 / monthly_income
        # Add small epsilon to avoid div by zero
        features_df['dti_ratio'] = (features_df['emi_outflow'] * 12) / (features_df['monthly_income'] + 1e-6)
        
        # credit_utilization = used_credit / total_credit
        features_df['credit_utilization'] = features_df['used_credit'] / (features_df['total_credit'] + 1e-6)
        
        # delinquency_flag
        features_df['delinquency_flag'] = (features_df['delinquency_count'] > 0).astype(int)
        
        return features_df
        
    def train_model(self, features_df):
        features = ['dti_ratio', 'credit_utilization', 'delinquency_flag', 'credit_score']
        X = features_df[features].fillna(0)
        y = features_df['default']
        model = LogisticRegression(max_iter=100)
        model.fit(X, y)
        features_df['probability'] = model.predict_proba(X)[:, 1]
        return features_df, model

    def cross_validate_model(self, features_df):
        from sklearn.model_selection import cross_val_score
        features = ['dti_ratio', 'credit_utilization', 'delinquency_flag', 'credit_score']
        X = features_df[features].fillna(0)
        y = features_df['default']
        scores = cross_val_score(LogisticRegression(), X, y, cv=5, scoring='roc_auc')
        return scores.mean()

    def evaluate_model(self, predictions):
        y_true = predictions['default']
        y_pred = predictions['probability']
        auc = roc_auc_score(y_true, y_pred)
        return auc

    def calculate_risk_scores(self, predictions):
        scored_df = predictions.copy()
        scored_df['risk_score'] = scored_df['probability']
        
        # Risk Buckets
        conditions = [
            (scored_df['risk_score'] > 0.7),
            (scored_df['risk_score'] > 0.4) & (scored_df['risk_score'] <= 0.7),
            (scored_df['risk_score'] <= 0.4)
        ]
        choices = ['High', 'Medium', 'Low']
        
        scored_df['risk_bucket'] = np.select(conditions, choices, default='Low')
        
        return scored_df[['borrower_id', 'risk_score', 'risk_bucket', 'probability']]
