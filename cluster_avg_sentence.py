import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


class ClusterAverageSentenceGenerator:
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize the generator with a BERT model.

        Args:
            model_name (str): Name of the BERT model to use
        """
        print(f"Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def get_bert_embedding(self, text):
        """
        Get BERT embedding for a single text.

        Args:
            text (str): Input text

        Returns:
            numpy.ndarray: BERT embedding vector
        """
        # Tokenize and encode the text
        inputs = self.tokenizer(text, return_tensors='pt',
                                truncation=True, padding=True,
                                max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding.squeeze()

    def get_batch_embeddings(self, texts, batch_size=32):
        """
        Get BERT embeddings for multiple texts in batches.

        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing

        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors='pt',
                                    truncation=True, padding=True,
                                    max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def find_closest_sentence(self, target_embedding, candidate_sentences, candidate_embeddings):
        """
        Find the sentence with embedding closest to the target embedding.

        Args:
            target_embedding (numpy.ndarray): Target embedding vector
            candidate_sentences (list): List of candidate sentences
            candidate_embeddings (numpy.ndarray): Embeddings of candidate sentences

        Returns:
            str: The closest sentence
        """
        # Calculate cosine similarities
        similarities = cosine_similarity([target_embedding], candidate_embeddings)[0]

        # Find the index of the most similar sentence
        closest_idx = np.argmax(similarities)

        return candidate_sentences[closest_idx], similarities[closest_idx]

    def process_excel_file(self, file_path, sentence_column='sentence', cluster_column='cluster'):
        """
        Process the Excel file and add average sentences for each cluster.

        Args:
            file_path (str): Path to the Excel file
            sentence_column (str): Name of the sentence column
            cluster_column (str): Name of the cluster column

        Returns:
            pandas.DataFrame: Updated dataframe with avg_sentence column
        """
        print(f"Reading Excel file: {file_path}")

        # Read the Excel file
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None

        # Validate required columns exist
        if sentence_column not in df.columns:
            print(f"Error: Column '{sentence_column}' not found in the Excel file.")
            print(f"Available columns: {list(df.columns)}")
            return None

        if cluster_column not in df.columns:
            print(f"Error: Column '{cluster_column}' not found in the Excel file.")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Remove rows with missing sentences or clusters
        df = df.dropna(subset=[sentence_column, cluster_column])

        print(f"Processing {len(df)} sentences across {df[cluster_column].nunique()} clusters...")

        # Get all unique sentences and their embeddings
        unique_sentences = df[sentence_column].unique().tolist()
        print("Computing embeddings for all unique sentences...")
        all_embeddings = self.get_batch_embeddings(unique_sentences)

        # Create a mapping from sentence to embedding
        sentence_to_embedding = dict(zip(unique_sentences, all_embeddings))

        # Initialize the avg_sentence column
        df['avg_sentence'] = None

        # Process each cluster
        unique_clusters = df[cluster_column].unique()

        for cluster_id in unique_clusters:
            print(f"\nProcessing cluster {cluster_id}...")

            # Get sentences in this cluster
            cluster_mask = df[cluster_column] == cluster_id
            cluster_sentences = df[cluster_mask][sentence_column].tolist()

            # Skip if cluster is empty or has only one sentence
            if len(cluster_sentences) == 0:
                continue
            elif len(cluster_sentences) == 1:
                print(f"Cluster {cluster_id}: Only 1 sentence, using itself as average")
                df.loc[cluster_mask, 'avg_sentence'] = cluster_sentences[0]
                continue

            # Handle special cases (like cluster -1 for unclustered sentences)
            if cluster_id == -1:
                print(f"Cluster {cluster_id}: Unclustered sentences ({len(cluster_sentences)} sentences)")
                print("Each sentence will be its own average")
                # For unclustered sentences, each sentence is its own average
                for idx, sentence in zip(df[cluster_mask].index, cluster_sentences):
                    df.loc[idx, 'avg_sentence'] = sentence
                continue

            # Get embeddings for sentences in this cluster
            cluster_embeddings = np.array([sentence_to_embedding[sent] for sent in cluster_sentences])

            # Calculate average embedding
            avg_embedding = np.mean(cluster_embeddings, axis=0)

            # Find the sentence closest to the average embedding
            closest_sentence, similarity = self.find_closest_sentence(
                avg_embedding, unique_sentences, all_embeddings
            )

            print(f"Cluster {cluster_id}: {len(cluster_sentences)} sentences")
            print(f"Average sentence (similarity: {similarity:.3f}): {closest_sentence[:100]}...")

            # Assign the average sentence to all rows in this cluster
            df.loc[cluster_mask, 'avg_sentence'] = closest_sentence

        return df

    def save_results(self, df, output_path):
        """
        Save the results to an Excel file.

        Args:
            df (pandas.DataFrame): Dataframe to save
            output_path (str): Output file path
        """
        print(f"\nSaving results to: {output_path}")
        df.to_excel(output_path, index=False)
        print("Results saved successfully!")


def main():
    """
    Main function to run the cluster average sentence generation.
    """
    # Configuration
    INPUT_FILE = "sentences_with_clusters.xlsx"
    OUTPUT_FILE = "sentences_with_clusters_updated.xlsx"
    SENTENCE_COLUMN = "sentence"  # Adjust if your column name is different
    CLUSTER_COLUMN = "cluster"  # Adjust if your column name is different

    try:
        # Initialize the generator
        generator = ClusterAverageSentenceGenerator()

        # Process the Excel file
        df_updated = generator.process_excel_file(
            INPUT_FILE,
            sentence_column=SENTENCE_COLUMN,
            cluster_column=CLUSTER_COLUMN
        )

        if df_updated is not None:
            # Save the results
            generator.save_results(df_updated, OUTPUT_FILE)

            # Display summary
            print(f"\n{'=' * 50}")
            print("SUMMARY")
            print(f"{'=' * 50}")
            print(f"Total sentences processed: {len(df_updated)}")

            # Count actual clusters (excluding -1 if it exists)
            clustered_data = df_updated[df_updated[CLUSTER_COLUMN] != -1] if -1 in df_updated[
                CLUSTER_COLUMN].values else df_updated
            unclustered_count = len(df_updated[df_updated[CLUSTER_COLUMN] == -1]) if -1 in df_updated[
                CLUSTER_COLUMN].values else 0

            print(f"Number of actual clusters: {clustered_data[CLUSTER_COLUMN].nunique()}")
            if unclustered_count > 0:
                print(f"Unclustered sentences: {unclustered_count}")
            print(f"Output saved to: {OUTPUT_FILE}")

            # Show a sample of results (excluding unclustered if they exist)
            print(f"\nSample results:")
            display_df = clustered_data if len(clustered_data) > 0 else df_updated
            sample_df = display_df.groupby(CLUSTER_COLUMN).first().reset_index().head(3)
            for _, row in sample_df.iterrows():
                print(f"Cluster {row[CLUSTER_COLUMN]}:")
                print(f"  Sample sentence: {row[SENTENCE_COLUMN][:80]}...")
                print(f"  Average sentence: {row['avg_sentence'][:80]}...")
                print()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()