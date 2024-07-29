import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data = pd.read_csv('heart.csv')
subset_data = data[['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'target']]
print(subset_data.head())

model = BayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('thalach', 'target'),
    ('exang', 'target'),
    ('oldpeak', 'target')
])

model.fit(subset_data, estimator=MaximumLikelihoodEstimator)
inference = VariableElimination(model)
evidence = { 'age': 50, 'sex': 1, 'cp': 2, 'thalach': 163, 'exang': 0, 'oldpeak': 0 }

result = inference.query(variables=['target'], evidence=evidence)
print(result)