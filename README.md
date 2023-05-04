# Crystal-structure










Crystal Structure Prediction




Submitted by :
Adarsh Kumar
200107004



















Git
Abstract
The goal of the study on crystal structure prediction was to investigate how machine learning techniques may be used to forecast the crystal structures of various materials. Three algorithms were used in the project: Decision Tree, Support Vector Machines (SVM), and Random Forest. Each algorithm was trained using a training set of data, and its performance was assessed using a separate validation set of data. The primary toolset utilised was the Python Scikit-learn framework, which offers a nice user interface for machine learning. The Random Forest method, followed by SVM and Decision Tree, was shown to be the most accurate at predicting crystal formations. The experiment shows how machine learning can increase the effectiveness and precision of material design and discovery, which has significant implications for the field of chemical engineering. However, the project has limitations, such as its exclusive focus on inorganic materials and its neglect of factors like temperature and pressure that could affect experiments. The concept might be expanded to include organic materials in the future, and the prediction models could incorporate experimental data.


Introduction :
The project's focus is on crystal structure prediction, a vital component of chemical engineering. It is crucial for the creation of new materials for a variety of applications since a material's crystal structure dictates its physical, chemical, and mechanical properties. The development of novel materials with qualities customised for particular applications can be facilitated by accurate crystal structure prediction, which can speed up the material discovery process and save time and resources.
Chemical engineering and materials science have recently placed a greater emphasis on artificial intelligence (AI) and machine learning (ML). They provide strong tools for complicated data analysis, complex system modelling, and process optimization, resulting in more practical and successful solutions. ML algorithms can be used to find patterns in data and predict crystal structures more effectively and correctly than conventional techniques.
The importance of this study rests in its capacity to accurately forecast crystal formations, which helps speed up the development of new materials. The dataset utilised in the project is a useful tool for academics and engineers working in the fields of chemical engineering since it contains details about the chemical make-up and crystal structures of various materials. The research also emphasises the significance of AI and ML in the field of chemical engineering by demonstrating the capability of ML algorithms to address challenging issues in these fields.







Methodology :
	In order to meet our requirements, the first step in the process is of data cleaning and pre processing.
	In our data set we have a total of 18 different features in which we have to predict “Lowest distortion” using other features. 

The first step in data cleaning is to find NaN values, which are not present in the given data set, but empty cells are present in different columns with the value “-”. 
From the figure, it’s clear that dashes are present in the “v(A)”, “v(B)”, 'τ' and “Lowest distortion” columns.

“Lowest Distortion” is going to be our target column. We can use entries with '-' in our testing phase, hence no need to think about handling this column's data, or we can also just remove these rows as the number of '-' is just 53, and we can't predict whether our prediction is correct or not corresponding to these columns.
In the case of 'τ' more than 50% of the entries are not present, which is why, rather than handling, we should drop this column; otherwise, this can affect our result.
The number of missing values is the same for the first and second elements. Indeed, if we check if these rows are the same, the result is positive: There are 1881 cases where both v(A) and v(B) are undefined. The best way to overcome this problem will be to use one-hot encoding and use “-” as a feature, As the number of dashes is significantly large, dropping or randomly defining the values is not going to be a good option.
Similarly we can use one-hot encoding for all the columns containing string values to convert them into numerical values.
And at last we’ll just drop the columns (Compound name, In literature) which are not going to be significantly useful in predicting our output.
Basically in our project we are using three machine learning algorithms: Random Forest, Support Vector Machines (SVM), and Decision Tree.
Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy. It was selected as the primary model due to its high accuracy, scalability, and ability to handle complex data.
SVM is a popular algorithm used in classification and regression tasks that works by finding a hyperplane in a high-dimensional space that best separates the data points into different classes. In this project, SVM was used as an alternative model to Random Forest.
Decision Tree is a simple model that uses a tree-like structure to make decisions based on a set of rules. It is a popular algorithm in machine learning due to its simplicity and interpretability. In this project, Decision Tree was used as a baseline model for comparison.
The implementation of each algorithm was done using the Scikit-learn library, which provides a user-friendly interface for machine learning in Python. Each algorithm was trained on a training set of data and evaluated on a separate validation set of data. The performance of each algorithm was measured using various evaluation metrics.	
Overall, the project utilized a combination of machine learning algorithms to predict the crystal structures of materials with high accuracy. The Random Forest algorithm was selected as the primary model, while SVM and Decision Tree were used as alternative models for comparison. 














Results :


	The crystal structure prediction project used several machine learning algorithms, including Random Forest, Support Vector Machine (SVM), and Decision Tree, to predict the crystal structures of inorganic materials. The performance of these algorithms was evaluated based on several metrics, including accuracy, precision, recall, and F1 score.


The Random Forest algorithm achieved the highest accuracy of 79%, followed by the SVM algorithm with an accuracy of 69% and the Decision Tree algorithm with an accuracy of 73%. These results indicate that the Random Forest algorithm was the most effective at predicting the crystal structures of the materials.
Additionally, the project highlights the significance of machine learning in the field of chemical engineering. By predicting crystal structures, machine learning algorithms can help identify materials with desirable properties, such as high thermal conductivity or hardness, for use in various applications. This can save time and resources in the materials discovery process, and potentially lead to the development of new materials with improved properties.






Conclusion :
The primary conclusions of the project on crystal structure prediction are that machine learning algorithms like Random Forest, SVM, and Decision Tree can predict the crystal structures of materials with high accuracy. SVM and Decision Tree were equally effective but not as accurate as the Random Forest algorithm for predicting crystal structures. These findings illustrate the potential for machine learning to enhance the effectiveness and precision of material design and discovery, which has major implications for the area of chemical engineering.
The project does, however, have some constraints that must be taken into account. One drawback is that the project only attempted to predict the crystal structures of inorganic materials, and when applied to biological materials, the algorithms may behave differently. Another drawback is that the project did not take into account how different experimental circumstances, including temperature and pressure, can affect crystal structure prediction.
Future work could concentrate on expanding the concept to include organic materials and putting experimental circumstances into the prediction models to solve these constraints. Additionally, the potential of other machine learning algorithms, such as neural networks, for crystal structure prediction could be investigated. Overall, the project's findings demonstrate the potential for machine learning to revolutionise material design and discovery and have significant ramifications for the discipline of chemical engineering.



References :
https://www.kaggle.com/datasets/sayansh001/crystal-structure-classification
https://pubs.acs.org/doi/10.1021/acs.jctc.2c00451
https://www.researchgate.net/publication/260211208_Crystal_Structure_Prediction_and_Its_Application_in_Earth_and_Materials_Sciences










