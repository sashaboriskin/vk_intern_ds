from train import fit
import pandas as pd

solution = pd.read_csv('data/sample_submission.csv')
preds = fit()
solution['score'] = preds

solution.to_csv('submission.csv', index=False)