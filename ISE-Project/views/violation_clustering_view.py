# views/violation_clustering_view.py

import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.decorators import log_execution, time_execution

class ViolationClusteringView:
    def __init__(self, data) -> None:
        self.data = data

    @log_execution
    @time_execution
    def render(self) -> None:
        st.title("K-Means Clustering nach Verstoßmustern")
        st.markdown("""
        **Ziel:** Restaurants nach ihren Verstoßmustern clustern, um ähnliche Muster zu identifizieren.
        """)
        df = self.data
        violation_matrix = df.groupby(['restaurant_name', 'violation_code']).size().unstack(fill_value=0)
        scaler = StandardScaler()
        violation_matrix_scaled = scaler.fit_transform(violation_matrix)
        k = st.slider("Wähle Anzahl der Cluster (k) für Violation-Clustering", min_value=2, max_value=15, value=4, key='violation_cluster_slider')
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(violation_matrix_scaled)
        cluster_labels = pd.Series(kmeans.labels_, index=violation_matrix.index, name="cluster_violation")
        cluster_df = pd.DataFrame(cluster_labels).reset_index()
        df_clustered = pd.merge(df, cluster_df, on='restaurant_name', how='left')
        df_clustered['cluster_violation'] = df_clustered['cluster_violation'].astype(str)
        st.markdown("### Beispieldaten (Cluster-Zugehörigkeit)")
        st.dataframe(df_clustered.head(20))
        pca = PCA(n_components=2, random_state=42)
        principal_comps = pca.fit_transform(violation_matrix_scaled)
        pca_df = pd.DataFrame(principal_comps, columns=["PC1", "PC2"], index=violation_matrix.index)
        pca_df['cluster_violation'] = cluster_labels.astype(str)
        pca_df['restaurant_name'] = pca_df.index
        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="cluster_violation",
            hover_data=["restaurant_name"],
            title=f"K-Means Clustering nach Verstoßmustern (k={k}), PCA-Visualisierung"
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        cluster_size = pca_df['cluster_violation'].value_counts().reset_index()
        cluster_size.columns = ['cluster_violation', 'count']
        fig_cluster_size = px.bar(
            cluster_size,
            x='cluster_violation',
            y='count',
            color='cluster_violation',
            title="Anzahl Restaurants pro Cluster (Violations-Based)",
            labels={'cluster_violation': 'Cluster', 'count': 'Anzahl Restaurants'}
        )
        st.plotly_chart(fig_cluster_size, use_container_width=True)
        st.markdown("""
        **Business Case:**
        - Restaurants mit ähnlichen Verstoßmustern können gezielt geschult werden.
        - Behörden können Maßnahmen zielgerichtet planen.
        """)