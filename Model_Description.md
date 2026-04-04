# Model Description
This is the document about the model I choose for the final project. This readme file will contains four parts, first, how we deal with the data, or we call it "embedding"; second, the model I choose for the clustering; third, the model for the prediction; forth, the metrics I use to evaluate the model.

### Dataset Review and Embedding Method
| Feature Name | Description | Data Type |
| :--- | :--- | :--- |
| **StudyHours** | Number of study hours per week | Numeric |
| **Attendance** | Percentage of classes attended | Numeric |
| **Resources** | Availability and use of academic resources (e.g., library, notes) | Categorical |
| **Extracurricular** | Participation in extracurricular activities | Binary (Yes/No) |
| **Motivation** | Self-reported motivation level | Numeric Scale |
| **Internet** | Access to the internet for study purposes | Binary (Yes/No) |
| **Gender** | Student’s gender | Categorical |
| **Age** | Age of student (18–30 years) | Numeric |
| **LearningStyle** | Preferred learning style (Visual, Auditory, Kinesthetic, etc.) | Categorical |
| **OnlineCourses** | Participation in online courses | Binary (Yes/No) |
| **Discussions** | Engagement in study group discussions or forums | Numeric Scale |
| **AssignmentCompletion** | Rate of completing assignments on time | Numeric Scale |
| **EduTech** | Usage of educational technology tools/platforms | Numeric Scale |
| **StressLevel** | Self-reported stress level | Numeric Scale |
| **ExamScore** | Score obtained in the main exam (Target 1) | Numeric |
| **FinalGrade** | Final course grade (Target 2) | Numeric |

Here, my first idea is to use **one-hot** encoding, because it's a very easy way to convert any kind of data to numeric data, but then I realize, there excist a very strong orthogonal characteristics after encoding, so the feature after encoding will be more independent. But according the this dataset, each feature have some correlationships with others, which means each features are not independent, we have to consider the latent information about each feature (e.g. syntax information), so here if I use one-hot encoding, we definitely will lose a lot of important information. So here, I plan to use one embedding method called **Entity Embedding**. Comparing with the one-hot encoding, entity embedding can learn the relationship between each feature, so, it can maps similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables. I think this embedding is very suit my final dataset.

### Model for Clustering
For clustering, we need to use both deep learning model and classical model. After we apply entity embedding to our dataset, I will choose **Variational AutoEncoder (VAE)** to be my deep learning model. There is a huge advantange that the VAE has compared with other encoder, that is, VAE can encodes continuous, probabilistic representation of that latent space. Here, since each features have correlationship, so I choose VAE be my deep learning model. After we finish train the VAE, I will drop the decoder part, because my goal is to do clustering, not generate new samples, so I will only need encoder and latent space. Then, I pass these latent space value to **K-means**, then I will obtain the final clusters.

The pipline of clustering shown as follows:
- Apply Entity Embedding to Dataset
- Apply Variational AutoEncoder
    - Train the entire Variational AutoEncoder
    - After finish training, drop Decoder
    - Only use Encoder to get the latent space
- Apple K-means

Variational AutoEncoder actually gives us the probability of the data, I think this is what we need for clustering, if two data point have the same characteristics, they should be very close and may be in the same cluster using the same label, in such situation, their mean should be very close, so this how I think VAE should be useful for clustering, but this is my assumption.

### Model for Prediction
For prediction, I will choose **Feature Tokenizer + Transformer (FT-Transformer)**. Actually, this FT-Transformer is a model combine the feature tokenizer and transformer together, the pipline of FT-Transformer is prety easy. First, we just give different weight to the feature, to convert them into a full, long token; second, pass this token to transformer; third, make predictaion.
$$\hat{y} = Linear(ReLU(LayerNorm(T^{[CLS]}_L)))$$
Here, $CLS$ reprsents the "classification token", or "output token". The reason I choose this model is after we do the cluster, we will have the cluster id, latent space data, and the original data, we can combine them together become a huge token, which actually along with FT-Transformer, and I think in this way, we do have enough data to capture the information and make a prediction.

But, I should pay attention to the limitation of FT-Transformer that requires more resources (both hardware and time) for training than simple models.


### Metrcis
For Clustering:
- **Silhouette Coefficient**: is computed as the difference between the average distance to points in the nearest cluster (separation) and the average distance to points in the same cluster (cohesion), divided by the maximum of these two values. Values range from −1 to 1, where higher values indicate more coherent clusters and negative values suggest potential misassignment.
- **Davies Bouldin Score**: is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score.
- **Calinski-Harabasz Index**: can be used to evaluate the model, where a higher Calinski-Harabasz score relates to a model with better defined clusters.

For Prediction:
- **$R^2$**: is the proportion of the variation in the dependent variable that is predictable from the independent variable(s)
- **RMSE (Root Mean Squared Error)**: is a frequently used measure of the distances between actual observed values and an estimation of them.

### References
[1] Guo, C. and Berkhahn, F., 2016. Entity embeddings of categorical variables. arXiv preprint arXiv:1604.06737.

[2] Kingma, D.P. and Welling, M., 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

[3] https://www.ibm.com/think/topics/variational-autoencoder#:~:text=Variational%20autoencoders%20(VAEs)%20are%20generative,other%20autoencoders%2C%20such%20as%20denoising.

[4] Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A., 2021. Revisiting deep learning models for tabular data. Advances in neural information processing systems, 34, pp.18932-18943.

[5] https://www.sciencedirect.com/topics/computer-science/silhouette-coefficient

[6] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html

[7] https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index

[8] https://en.wikipedia.org/wiki/Coefficient_of_determination

[9] https://en.wikipedia.org/wiki/Root_mean_square_deviation