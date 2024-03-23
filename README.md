# Example code for Feature Selection 

1. Random Feature
  - Similar to Boruta algorithm
  - The difference from the Boruta algorithm is that while Boruta creates variables by randomly mixing the values ​​of existing features,
  - This algorithm calculates feature importance after adding several random features generated with random numbers(uniform(0, 1)).
  - After calculating feature importance, the method of removing variables with lower variable importance than randomly added variables is the same.

2. Genetic Algorithm
  - Feature selection method was implemented using genetic algorithm.
  - The algorithm was constructed by generating the use or non-use (0, 1) of each variable as a gene, calculating the fitness, selecting only the top K, and then crossing them.
  - The roulette wheel method is used to select genes in each generation.
  - For Score of Fitness, scikit_learn's cross_val_score function was used.

3. Boruta
   - TBD

99. Data Source
  - Kaggle: Santander Customer Satisfaction
  - Data shape: (76020, 371)
  - Binary classification(0, 1) and Class imblance(0: 58410, 1: 2406)
  - https://www.kaggle.com/c/santander-customer-satisfaction
