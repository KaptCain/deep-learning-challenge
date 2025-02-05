Module 21 Deep Learning Challenge
Created by Alec Collins
I tried so many different variations of epochs, batch sizes, and number of layers, but was never able to reach a 75% accuracy score.


Alphabet Soup Charity Deep Learning Model Analysis
Overview of the Analysis

The purpose of this analysis is to develop a deep learning model to predict whether applicants for funding from Alphabet Soup Charity will be successful. By training a neural network on historical application data, we aim to classify future applicants based on their likelihood of receiving funding. The model is expected to improve the efficiency of decision-making for the charity, ensuring that resources are allocated to the most promising applicants.
Results
Data Preprocessing

    What variable(s) are the target(s) for your model?
        The target variable for the model is IS_SUCCESSFUL, which indicates whether an applicant received funding.

    What variable(s) are the features for your model?
        The features include all columns related to applicant characteristics, such as:
            APPLICATION_TYPE
            AFFILIATION
            CLASSIFICATION
            USE_CASE
            ORGANIZATION
            STATUS
            INCOME_AMT
            SPECIAL_CONSIDERATIONS
            ASK_AMT

    What variable(s) should be removed from the input data because they are neither targets nor features?
        The columns EIN and NAME were removed because they are identifiers that do not contribute to predicting the target variable.

Compiling, Training, and Evaluating the Model

    How many neurons, layers, and activation functions did you select for your neural network model, and why?
        The model consisted of:
            Input Layer: Matching the number of input features.
            Hidden Layers: 2 layers with:
                First hidden layer: 80 neurons, ReLU activation.
                Second hidden layer: 30 neurons, ReLU activation.
            Output Layer: 1 neuron with a sigmoid activation function for binary classification.
        Why?
            The ReLU activation was chosen for hidden layers to introduce non-linearity and avoid the vanishing gradient problem.
            The sigmoid activation was chosen for the output layer because it is suited for binary classification.

    Were you able to achieve the target model performance?
        No, the model did not achieve an accuracy significantly higher than 75%. Despite optimizations, it remained below expectations.

    What steps did you take in your attempts to increase model performance?
        Several approaches were used to improve model performance:
            Adjusting the number of layers and neurons to find the optimal architecture.
            Implementing dropout layers to prevent overfitting.
            Trying different activation functions (e.g., ReLU and tanh) to test their impact on training performance.
            Applying additional preprocessing techniques like feature scaling using StandardScaler.
            Tuning learning rates and incorporating ReduceLROnPlateau to adjust the learning rate dynamically.

Summary
Overall Results of the Deep Learning Model

The deep learning model was able to classify successful applicants but did not reach the desired level of accuracy. The model performed well on training data but did not generalize as effectively on test data, suggesting that further hyperparameter tuning or feature engineering is needed.
Alternative Model Recommendation

A potential alternative model is XGBoost (Extreme Gradient Boosting). XGBoost is highly effective for structured/tabular data and can outperform deep learning models when dealing with categorical and numerical data.
Why Use XGBoost Instead?

    It handles missing data and categorical variables better.
    It trains faster and requires less computational power than deep learning models.
    It is less prone to overfitting with proper tuning (e.g., using tree depth, learning rate, and regularization techniques).
    It provides feature importance scores, making interpretation easier.

Using XGBoost, we may be able to achieve higher accuracy and better interpretability compared to a deep learning approach.