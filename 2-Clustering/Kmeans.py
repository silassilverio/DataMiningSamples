# Implementation of Kmeans from scratch and using sklearn
# Loading the required modules
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Defining our kmeans function from scratch


def KMeans_scratch(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    # Randomly choosing Centroids
    centroids = x[idx, :]  # Step 1

    # finding the distance between centroids and all the data points
    distances = cdist(x, centroids, 'euclidean')  # Step 2

    # Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances])  # Step 3

    # Repeating the above steps for a defined number of iterations
    # Step 4
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Updated Centroids

        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points


def plot_samples(projected, labels, title):
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i, 0], projected[labels == i, 1], label=i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.title(title)


def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")


def main():
    # Load dataset Digits
    input_file = '0-Datasets/hcc-dataClear.txt'
    names = ['Gender', 'Symptoms', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B e Antigen', 'Hepatitis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis', 'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity', 'Hemochromatosis', 'Arterial Hypertension', 'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus', 'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly', 'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metastasis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day', 'Packs of cigarets per year',
             'Performance Status', 'Encefalopathy degree', 'Ascites degree', 'International Normalised Ratio', 'Alpha-Fetoprotein (ng/mL)', 'Haemoglobin (g/dL)', 'Mean Corpuscular Volume (fl)', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)', 'Total Bilirubin(mg/dL)', 'Alanine transaminase (U/L)', 'Aspartate transaminase (U/L)', 'Gamma glutamyl transferase (U/L)', 'Alkaline phosphatase (U/L)', 'Total Proteins (g/dL)', 'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)', 'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)', 'Class']
    features = ['Age at diagnosis', 'Grams of Alcohol per day', 'Packs of cigarets per year',
                'Performance Status', 'Encefalopathy degree', 'Ascites degree', 'International Normalised Ratio', 'Alpha-Fetoprotein (ng/mL)', 'Haemoglobin (g/dL)', 'Mean Corpuscular Volume (fl)', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)', 'Total Bilirubin(mg/dL)', 'Alanine transaminase (U/L)', 'Aspartate transaminase (U/L)', 'Gamma glutamyl transferase (U/L)', 'Alkaline phosphatase (U/L)', 'Total Proteins (g/dL)', 'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)', 'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)', 'Class']
    target = 'Class'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names=names)  # Nome das colunas
    k = 2
    x = df.loc[:, features].values
    y = df.loc[:, [target]].values

    x = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data=x, columns=features)
    normalized2Df = pd.concat([normalized2Df, df[[target]]], axis=1)
    ShowInformationDataFrame(normalized2Df, "Dataframe Min-Max Normalized")

    # Transform the data using PCA
    pca = PCA()
    projected = pca.fit_transform(x)
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")

    # Applying our kmeans function from scratch
    labels = KMeans_scratch(projected, k, 100)

    # Visualize the results
    plot_samples(projected, labels, 'Clusters Labels KMeans from scratch')

    # Applying sklearn kemans function
    kmeans = KMeans(n_clusters=k, random_state=0).fit(projected)
    print(kmeans.inertia_)
    centers = kmeans.cluster_centers_
    score = silhouette_score(projected, kmeans.labels_)
    print("For n_clusters = {}, silhouette score is {})".format(k, score))

    # Visualize the results sklearn
    plot_samples(projected, kmeans.labels_,
                 'Clusters Labels KMeans from sklearn')

    plt.show()


if __name__ == "__main__":
    main()
