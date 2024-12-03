This project focuses on Sanction Screening Using Deep Neural Networks. 

This study investigates the use of deep neural networks (DNNs) to automate the sanction screening process, utilizing various architectural approaches to predict whether an entity is sanctioned based on a variety of features. 

These features include categorical variables such as entity names, sanction types, and geographic details, along with temporal information from sanction start and expiry dates, as well as textual data from additional fields. 

The first model explores a sequential DNN architecture using basic fully con- nected layers, where activation functions like ReLU employed to model complex relationships between the input features and the target variable (sanction status). Subsequently, the model is refined by incorporating skip connections to introduce more complex dependencies between layers, alongside dropout and batch nor- malization for regularization. 

To further enhance the robustness of the model, a second approach uses a deep neural network with LeakyReLU and ELU activations, exploring the use of non-linear activation functions to combat problems such as the problem of disappearance of the gradient and improve convergence. Addi- tionally, the architecture incorporates batch normalization and dropout layers for better regularization and to reduce overfitting. 

The key features used for the predictions include the sanction entity, the sanction type, the vessel ID, the start and expiration dates of the sanction, the address, the city, the state and other geographical and temporal details. Pre-processing steps involve encoding categorical variables, imputing missing data, and transforming date features into actionable attributes like sanction duration and age. 

Performance evaluation of these models is conducted using standard classification metrics, including accuracy, precision, recall, and F1-score, with the goal of achieving a reliable and interpretable solution for sanction screening tasks. The data collected from USA Government agency OFAC for this studies. Ultimately, the study demonstrates the feasibility of using deep learning techniques for sanction detection, offering insights into how different DNN architectures can be optimized for real-world applications in compliance and risk management. These models show promise in reducing manual screening workloads and enhancing the accuracy of sanction-related decisions in various industries.
