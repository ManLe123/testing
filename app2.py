import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.title("Spectral Clustering")
st.write("This app performs spectral clustering on the provided dataset.")

#Take data from 2014-2016
data_file = st.file_uploader("Upload CSV", type=["csv"])
if data_file is not None:
    data = pd.read_csv(data_file)
    data_selected_years = data[data['match_year'].isin([2016, 2015, 2014])]
    
    #Encode 'winning_team' (home win = 1, draw = 0, away win = -1)
    data_selected_years['winning_team'] = data_selected_years['winning_team'].map({'Home': 1, 'Draw': 0, 'Away': -1})
    
    features1 = data_selected_years[['B365H', 'B365D', 'B365A','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'winning_team']]
    
    from sklearn.ensemble import IsolationForest
    # Step 1: Initialize the Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    # Step 2: Fit the model to your data
    iso_forest.fit(features1)
    # Step 3: Predict which points are outliers (-1 for outliers, 1 for inliers)
    outliers = iso_forest.predict(features1)
    # Step 4: Create a new DataFrame excluding the outliers
    data_no_outliers = features1[outliers == 1]
    
    features2 = ['B365H', 'B365D', 'B365A','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
    
    X = data_no_outliers[features2]
    y = data_no_outliers['winning_team']
    
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X_resampled)
    
    from sklearn.decomposition import PCA
    import numpy as np
    # Plotting the cumulative variance ratio can help decide the number of components
    import matplotlib.pyplot as plt
    
    pca = PCA()
    pca.fit(data_scaled)
    
    # Getting the cumulative variance
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)*100
    print(cumulative_variance_ratio)
    
    # How many PCs explain 95% of the variance?
    k = np.argmax(cumulative_variance_ratio>95)
    print("Number of components explaining 95% variance: "+ str(k))
    #print("\n")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=[10, 5])  # Create a figure and an axis
    ax.set_title('Cumulative Explained Variance explained by component')
    ax.set_ylabel('Cumulative Explained variance (%)')
    ax.set_xlabel('Principal components')
    
    # Add vertical and horizontal lines
    ax.axvline(x=k, color="k", linestyle="--")
    ax.axhline(y=95, color="r", linestyle="--")
    
    # Plot the cumulative variance ratio
    ax.plot(cumulative_variance_ratio)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

 #   st.write("Enter the n_components")
  #  number = st.number_input("Enter a number", min_value=0, max_value=5, key="unique_key1", format="%d")

    pca = PCA(n_components=1)
    data_pca = pca.fit_transform(data_scaled)
  
    
    import plotly.express as px
    from sklearn.cluster import SpectralClustering , KMeans
    # Calculate the inertia for different cluster sizes
    inertia = []
    for i in range(1, 10):
        cluster = KMeans(n_clusters=i)
        cluster.fit(data_pca)
        inertia.append(cluster.inertia_)
    
    # Create the Elbow graph using Plotly
    fig = px.line(x=range(1, 10), y=inertia, 
                  title="Elbow Graph for Spectral Clustering",
                  labels={"x": "Number of Clusters", "y": "Inertia"})
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)

  #  st.write("Enter number of clusters")
#    number2 = st.number_input("Enter a number", min_value=1, max_value=5, key="unique_key2", format="%d")
    
    # Perform Spectral Clustering
    spectral_clustering = SpectralClustering(n_clusters=3, affinity='rbf', gamma=10, random_state=42)
    clusters = spectral_clustering.fit_predict(data_pca)

    test_index = range(len(data_pca))
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis
    scatter = ax.scatter(range(len(test_index)), data_pca[test_index, 0], 
                         c=clusters[test_index], cmap='viridis')
    
    # Add labels, title, and colorbar
    ax.set_title('Spectral Clustering based on Betting Odds')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('PCA Component')
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('Cluster')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    from sklearn.metrics import davies_bouldin_score
    
    # Compute Davies-Bouldin Index using PCA-transformed data
    db_score = davies_bouldin_score(data_pca, clusters)
    st.write(f'Davies-Bouldin Index (on PCA-transformed data): {db_score}')
    
    from sklearn.metrics import calinski_harabasz_score
    
    # Compute Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(data_pca, clusters)
    st.write(f'Calinski-Harabasz Index: {ch_score}')
    
    y_resampled_df = y_resampled.to_frame()
    print(y_resampled_df.columns)
    
    # Add cluster assignments as a new column to the DataFrame.
    X_resampled['Spectral_Cluster'] = clusters
    
    # Analyze clusters with respect to prediction quality
    # Loop through each cluster and display the prediction quality distribution in Streamlit
    for cluster_id in range(3):
        cluster_data = y_resampled_df[X_resampled['Spectral_Cluster'] == cluster_id]
        quality_distribution = cluster_data['winning_team'].value_counts()
        
        # Display cluster prediction quality distribution in Streamlit
        st.write(f'Cluster {cluster_id} Prediction Quality Distribution:')
        st.write(quality_distribution)
        st.write("")  # Adds a space between clusters
        
        # Visualization of Clusters vs. Actual Outcomes
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis
    scatter = ax.scatter(range(len(test_index)), data_pca[test_index, 0], 
                         c=y_resampled[test_index], cmap='viridis', marker='o', 
                         alpha=0.5, label='Actual Outcomes')
    
    # Add title, labels, and colorbar
    ax.set_title('PCA of Data with Actual Outcomes')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('PCA Component')
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('Actual Outcome')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    import streamlit as st
    import matplotlib.pyplot as plt
    
    # Assuming `test_index`, `data_pca`, and `clusters` are defined variables
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis
    scatter = ax.scatter(range(len(test_index)), data_pca[test_index, 0], 
                         c=clusters[test_index], cmap='viridis', marker='o', 
                         alpha=0.5, label='Clusters')
    
    # Add title, labels, and colorbar
    ax.set_title('PCA of Data with Clustering Results')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('PCA Component')
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('Cluster')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file.")



 
