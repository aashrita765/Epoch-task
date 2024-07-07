TASK 1:

This code performs K-means clustering on geographical data of pincodes in Telangana, India, using latitude and longitude to identify optimal clusters.

1. Data Preprocessing:
   - Loading the data, filtering for Telangana state, removing duplicates and NaNs, and converting latitude and longitude to float.

2. Visualization:
   - Plotting the original geographical points to visualize their distribution.

3. K-means Clustering:
   - Initializing random centroids.
   - Assigning labels to each data point based on the nearest centroid.
   - Updating centroids to be the mean of assigned points.
   - Iterating until centroids stabilize or reach max iterations.

4. Optimal Number of Clusters:
   - Elbow Method: Plotting Within-Cluster Sum of Squares (WCSS) for `k` from 1 to `max_k`. The "elbow" point suggests the optimal `k`.
   - Silhouette Method: Calculating silhouette scores for `k` from 2 to `max_k`. The highest score indicates the optimal `k`.

5. Final Clustering and Visualization:
   - I Chose k=5 based on the above methods.
   - Performed K-means clustering and visualize the clusters on a scatter plot.

Results:
- Identified 5 clusters of geographical locations in Telangana.
- Clusters are visualized with different colors, helping in understanding regional groupings.

These clusters can be used for applications like market segmentation and regional planning.

TASK 2:

The methods involves in the task 2 i.e recognising the text from the data which is trained are mentioned below:

Data Preprocessing:
1. Loading Dataset:
   - Loading and Cleaning: The `alphabets_28x28.csv` file is loaded. Non-numeric values are coerced to NaN and rows with NaN values are dropped to ensure clean data. All values are converted to float.
   - **Visualization**: The first 5 samples of handwritten alphabet images are displayed using Matplotlib to visually inspect the data.

2. Sentiment Analysis Dataset:
   - Loading and Cleaning: The sentiment analysis dataset is loaded from `sentiment_analysis_dataset.csv`. A `clean_text` function is used to convert text to lowercase and remove non-alphabetic characters, preparing it for model training.

Model Training:

1. CNN for Alphabet Classification:
   - A Convolutional Neural Network (CNN) is defined with layers for convolution, max pooling, flattening, and dense layers, designed to classify 28x28 pixel images of handwritten alphabets into 26 classes (A-Z).
   - The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. (Note: The actual training steps for the CNN are placeholders and need data preparation.)

2. Naive Bayes for Sentiment Analysis:
   - Text Vectorization: The cleaned text data is converted into word count vectors using `CountVectorizer`.
   - Model Training: The dataset is split into training and validation sets. A Multinomial Naive Bayes model is trained on the training data.
   - Model Evaluation: The model is evaluated on the validation set, achieving an accuracy of approximately 66.67%.

Inferences and Results:
- Alphabet Dataset: The preprocessing ensures clean and numeric data, and the visualized samples help confirm the data's integrity.
- Sentiment Analysis: The text cleaning process prepares the data for effective vectorization and model training. The Naive Bayes model achieves moderate accuracy, indicating room for improvement, potentially through more sophisticated text processing or model tuning.

This code sets up the foundation for both handwritten text recognition and sentiment analysis, with the sentiment analysis part being more complete and yielding initial accuracy results. The CNN part for alphabet classification needs further data preparation and training steps to be fully operational.
