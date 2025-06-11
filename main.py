import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import openpyxl

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    from sklearn.cluster import DBSCAN

    HDBSCAN_AVAILABLE = False
    print("⚠️  HDBSCAN not available. Using DBSCAN from sklearn as fallback.")

import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import spacy
from tqdm import tqdm
import time
from collections import Counter
import re
import warnings

warnings.filterwarnings('ignore')

# EMPATHY STRATEGY NAMES MAPPING
EMPATHY_STRATEGY_NAMES = {
    -1: "General Sympathy",
    0: "Detailed Validation",
    1: "Acknowledging Difficulty",
    2: "Offering Support",
    3: "Recognizing Deep Pain",
    4: "Validating Feelings",
    5: "Celebrating Joy",
    6: "Recognizing Strength",
    7: "Encouraging Self-Care",
    8: "Offering Wisdom",
    9: "Finding Silver Linings",
    10: "Inspiring Hope"
}


class SentenceClusteringAnalyzer:
    def __init__(self, masking_strategy='combined'):
        """
        Initialize the analyzer with masking strategy

        masking_strategy options:
        - 'none': No masking
        - 'subjects': Mask grammatical subjects
        - 'entities': Mask named entities (people, places, orgs, etc.)
        - 'pronouns': Normalize pronouns
        - 'combined': Mask entities + normalize pronouns (RECOMMENDED)
        - 'full': All masking strategies combined
        """
        self.model = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.cluster_labels = None
        self.sentences_df = None
        self.responses_with_clusters = None
        self.masking_strategy = masking_strategy
        self.nlp = None

        # Initialize spaCy if needed for masking
        if masking_strategy != 'none':
            try:
                self.nlp = spacy.load('en_core_web_sm')
                print(f"✅ Loaded spaCy model for masking strategy: {masking_strategy}")
            except OSError:
                print("⚠️  spaCy model 'en_core_web_sm' not found. Falling back to NLTK-based masking.")
                print("   Install with: python -m spacy download en_core_web_sm")
                self.nlp = None

    def create_tsne_visualization(self, embeddings, perplexity=30, random_state=42):
        """Create 2D t-SNE embeddings specifically for visualization"""
        print(f"Creating t-SNE visualization (perplexity={perplexity})...")

        # Limit to reasonable sample size for t-SNE (it's computationally expensive)
        if len(embeddings) > 10000:
            print(f"⚠️  Large dataset ({len(embeddings)} points). Consider sampling for t-SNE.")

        # Adjust perplexity if needed
        max_perplexity = min(50, (len(embeddings) - 1) // 3)
        perplexity = min(perplexity, max_perplexity)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            verbose=1 if len(embeddings) > 1000 else 0
        )

        tsne_embeddings = tsne.fit_transform(embeddings)
        return tsne_embeddings

    def mask_sentence(self, sentence):
        """Apply masking strategy to a sentence"""
        if self.masking_strategy == 'none':
            return sentence

        masked_sentence = sentence

        # Use spaCy if available, otherwise fall back to NLTK
        if self.nlp is not None:
            masked_sentence = self._mask_with_spacy(masked_sentence)
        else:
            masked_sentence = self._mask_with_nltk(masked_sentence)

        return masked_sentence

    def _mask_with_spacy(self, sentence):
        """Use spaCy for more accurate masking with NLP models"""
        doc = self.nlp(sentence)
        tokens = []

        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                tokens.append(token.text)
                continue

            # Mask named entities using spaCy's NER
            if self.masking_strategy in ['entities', 'combined', 'full'] and token.ent_type_:
                entity_mapping = {
                    'PERSON': '[PERSON]',
                    'ORG': '[ORGANIZATION]',
                    'GPE': '[LOCATION]',
                    'LOC': '[LOCATION]',
                    'EVENT': '[EVENT]',
                    'PRODUCT': '[PRODUCT]',
                    'WORK_OF_ART': '[WORK]',
                    'LAW': '[LAW]',
                    'LANGUAGE': '[LANGUAGE]',
                    'NORP': '[GROUP]',  # Nationalities, religious groups
                    'FAC': '[FACILITY]'
                }

                if token.ent_type_ in entity_mapping:
                    # Only mask if we're at the beginning of the entity span
                    if token.ent_iob_ == 'B':
                        tokens.append(entity_mapping[token.ent_type_])
                    # Skip other tokens in the entity (ent_iob_ == 'I')
                    elif token.ent_iob_ == 'I':
                        continue
                    else:
                        tokens.append(token.text)
                else:
                    tokens.append(token.text)
                continue

            # Normalize pronouns using spaCy's POS and lemma
            if self.masking_strategy in ['pronouns', 'combined', 'full'] and token.pos_ == 'PRON':
                pronoun_mapping = {
                    # First person
                    'i': '[I]', 'me': '[I]', 'my': '[I]', 'mine': '[I]', 'myself': '[I]',
                    # Second person
                    'you': '[YOU]', 'your': '[YOU]', 'yours': '[YOU]', 'yourself': '[YOU]', 'yourselves': '[YOU]',
                    # Third person singular
                    'he': '[HE/SHE]', 'him': '[HE/SHE]', 'his': '[HE/SHE]', 'himself': '[HE/SHE]',
                    'she': '[HE/SHE]', 'her': '[HE/SHE]', 'hers': '[HE/SHE]', 'herself': '[HE/SHE]',
                    'it': '[IT]', 'its': '[IT]', 'itself': '[IT]',
                    # Third person plural
                    'they': '[THEY]', 'them': '[THEY]', 'their': '[THEY]', 'theirs': '[THEY]', 'themselves': '[THEY]',
                    # First person plural
                    'we': '[WE]', 'us': '[WE]', 'our': '[WE]', 'ours': '[WE]', 'ourselves': '[WE]'
                }

                lower_token = token.text.lower()
                if lower_token in pronoun_mapping:
                    tokens.append(pronoun_mapping[lower_token])
                else:
                    tokens.append('[PRONOUN]')  # Fallback for other pronouns
                continue

            # Mask grammatical subjects using spaCy's dependency parsing
            if self.masking_strategy in ['subjects', 'full'] and token.dep_ in ['nsubj', 'nsubjpass', 'csubj',
                                                                                'csubjpass']:
                tokens.append('[SUBJECT]')
                continue

            # Additional masking for empathy-specific terms if using 'full' strategy
            if self.masking_strategy == 'full':
                # Mask emotion words
                emotion_words = {'happy', 'sad', 'angry', 'fear', 'joy', 'love', 'hate', 'worry', 'excited', 'nervous',
                                 'calm', 'upset', 'glad', 'mad', 'scared', 'pleased', 'disappointed', 'frustrated',
                                 'content', 'anxious'}
                if token.lemma_.lower() in emotion_words:
                    tokens.append('[EMOTION]')
                    continue

                # Mask mental state verbs
                mental_verbs = {'think', 'feel', 'believe', 'know', 'understand', 'realize', 'imagine', 'assume',
                                'suppose', 'consider', 'wonder', 'doubt', 'hope', 'expect', 'remember', 'forget'}
                if token.lemma_.lower() in mental_verbs:
                    tokens.append('[MENTAL_STATE]')
                    continue

                # Mask intensity adverbs
                intensity_adverbs = {'very', 'really', 'extremely', 'quite', 'pretty', 'somewhat', 'rather', 'fairly',
                                     'incredibly', 'absolutely', 'totally', 'completely', 'entirely'}
                if token.lemma_.lower() in intensity_adverbs and token.pos_ == 'ADV':
                    tokens.append('[INTENSITY]')
                    continue

            # Keep original token
            tokens.append(token.text)

        return ' '.join(tokens)

    def _mask_with_nltk(self, sentence):
        """Enhanced fallback masking using NLTK with better NLP processing"""
        # Download required NLTK data
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

        tokens = word_tokenize(sentence)

        # Enhanced entity masking with NLTK
        if self.masking_strategy in ['entities', 'combined', 'full']:
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)

            masked_tokens = []
            i = 0
            for chunk in chunks:
                if isinstance(chunk, Tree):
                    # This is a named entity
                    entity_type = chunk.label()
                    entity_mapping = {
                        'PERSON': '[PERSON]',
                        'ORGANIZATION': '[ORGANIZATION]',
                        'GPE': '[LOCATION]',
                        'LOCATION': '[LOCATION]',
                        'FACILITY': '[FACILITY]',
                        'GSP': '[LOCATION]'
                    }

                    if entity_type in entity_mapping:
                        masked_tokens.append(entity_mapping[entity_type])
                    else:
                        masked_tokens.append('[ENTITY]')
                    i += len(chunk)
                else:
                    # Regular token - apply other masking if needed
                    if i < len(tokens):
                        token = tokens[i]
                        token = self._apply_other_nltk_masking(token,
                                                               pos_tags[i] if i < len(pos_tags) else (token, 'NN'))
                        masked_tokens.append(token)
                        i += 1

            tokens = masked_tokens

        elif self.masking_strategy in ['pronouns', 'combined', 'full']:
            # Enhanced pronoun masking
            tokens = [self._apply_other_nltk_masking(token, None) for token in tokens]

        return ' '.join(tokens)

    def _apply_other_nltk_masking(self, token, pos_tag_tuple):
        """Apply non-entity masking using NLTK"""
        if self.masking_strategy in ['pronouns', 'combined', 'full']:
            # Enhanced pronoun mapping
            pronoun_mapping = {
                # First person
                'i': '[I]', 'me': '[I]', 'my': '[I]', 'mine': '[I]', 'myself': '[I]',
                # Second person
                'you': '[YOU]', 'your': '[YOU]', 'yours': '[YOU]', 'yourself': '[YOU]', 'yourselves': '[YOU]',
                # Third person singular
                'he': '[HE/SHE]', 'him': '[HE/SHE]', 'his': '[HE/SHE]', 'himself': '[HE/SHE]',
                'she': '[HE/SHE]', 'her': '[HE/SHE]', 'hers': '[HE/SHE]', 'herself': '[HE/SHE]',
                'it': '[IT]', 'its': '[IT]', 'itself': '[IT]',
                # Third person plural
                'they': '[THEY]', 'them': '[THEY]', 'their': '[THEY]', 'theirs': '[THEY]', 'themselves': '[THEY]',
                # First person plural
                'we': '[WE]', 'us': '[WE]', 'our': '[WE]', 'ours': '[WE]', 'ourselves': '[WE]'
            }

            lower_token = token.lower()
            if lower_token in pronoun_mapping:
                return pronoun_mapping[lower_token]

        # Enhanced masking for 'full' strategy
        if self.masking_strategy == 'full':
            lower_token = token.lower()

            # Emotion words
            emotion_words = {'happy', 'sad', 'angry', 'fear', 'joy', 'love', 'hate', 'worry', 'excited', 'nervous',
                             'calm', 'upset', 'glad', 'mad', 'scared', 'pleased', 'disappointed', 'frustrated',
                             'content', 'anxious'}
            if lower_token in emotion_words:
                return '[EMOTION]'

            # Mental state verbs
            mental_verbs = {'think', 'feel', 'believe', 'know', 'understand', 'realize', 'imagine', 'assume', 'suppose',
                            'consider', 'wonder', 'doubt', 'hope', 'expect', 'remember', 'forget'}
            if lower_token in mental_verbs:
                return '[MENTAL_STATE]'

            # Intensity adverbs (check POS tag if available)
            intensity_adverbs = {'very', 'really', 'extremely', 'quite', 'pretty', 'somewhat', 'rather', 'fairly',
                                 'incredibly', 'absolutely', 'totally', 'completely', 'entirely'}
            if lower_token in intensity_adverbs:
                if pos_tag_tuple is None or pos_tag_tuple[1].startswith('RB'):  # Adverb POS tags
                    return '[INTENSITY]'

        return token

    def load_and_prepare_data(self, csv_file, chosen_studies):
        """Load CSV data and prepare sentences with metadata"""
        print("🔹 Step 1: Loading and preparing data...")

        # Download necessary NLTK tokenizers
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

        # Load sentence transformer model
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Read CSV data
        data = pd.read_csv(csv_file, sep=',')

        # Filter data for chosen studies
        filtered_data = data[data['StudyNum'].isin(chosen_studies)]
        print(f"Total number of responses loaded: {len(filtered_data)}")

        # Prepare sentence records with metadata
        sentence_records = []

        print("Splitting responses into sentences...")
        for idx, row in tqdm(filtered_data.iterrows(), total=len(filtered_data), desc="Processing responses"):
            response = row['Response']
            story = row['Story']
            study_val = row['StudyNum']
            sub_id = row['SubID']

            if pd.notnull(response):
                sentences = sent_tokenize(response)
                for sent_idx, sent in enumerate(sentences):
                    sentence_records.append({
                        "original_row_index": idx,
                        "sentence_index_in_response": sent_idx,
                        "sentence": sent,
                        "Story": story,
                        "StudyNum": study_val,
                        "SubID": sub_id,
                        "original_response": response
                    })

        self.sentences_df = pd.DataFrame(sentence_records)
        print(f"Total number of sentences: {len(self.sentences_df)}")

        # Apply masking if specified
        if self.masking_strategy != 'none':
            print(f"🔹 Applying masking strategy: {self.masking_strategy}")
            masked_sentences = []
            for sentence in tqdm(self.sentences_df['sentence'], desc="Masking sentences"):
                masked_sentences.append(self.mask_sentence(sentence))

            # Add masked sentences to dataframe
            self.sentences_df['masked_sentence'] = masked_sentences

            # Save examples for inspection
            print("\n📝 Masking examples:")
            for i in range(min(3, len(self.sentences_df))):
                print(f"Original:  {self.sentences_df.iloc[i]['sentence'][:80]}...")
                print(f"Masked:    {self.sentences_df.iloc[i]['masked_sentence'][:80]}...")
                print("-" * 50)

        return self.sentences_df

    def create_embeddings(self):
        """Create sentence embeddings using masked sentences if available"""
        print("🔹 Step 2: Creating embeddings...")

        # Use masked sentences if available, otherwise original sentences
        if 'masked_sentence' in self.sentences_df.columns:
            sentences_to_embed = self.sentences_df["masked_sentence"].tolist()
            print("Using masked sentences for embedding...")
        else:
            sentences_to_embed = self.sentences_df["sentence"].tolist()
            print("Using original sentences for embedding...")

        self.embeddings = self.model.encode(sentences_to_embed, show_progress_bar=True)

        print(f"Embeddings shape: {self.embeddings.shape}")

        # Save embeddings
        np.save('embeddings.npy', self.embeddings)
        print("Embeddings saved to 'embeddings.npy'.")

        return self.embeddings

    def run_fixed_configuration(self, n_components=15, n_neighbors=3, min_dist=0.001):
        """Run clustering with your specific configuration"""
        print(f"🔹 Step 3: Running UMAP with fixed configuration...")
        print(f"   Dimensions: {n_components}")
        print(f"   Neighbors: {n_neighbors}")
        print(f"   Min distance: {min_dist}")

        # UMAP reduction with your configuration
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )

        self.reduced_embeddings = umap_reducer.fit_transform(self.embeddings)
        print(f"✅ UMAP reduction complete. Shape: {self.reduced_embeddings.shape}")

        # Apply optimized clustering
        print("🔹 Step 4: Finding optimal clustering parameters...")
        optimal_params, optimal_labels = self.find_optimal_clustering_parameters(self.reduced_embeddings)

        if optimal_params is not None and optimal_labels is not None:
            self.cluster_labels = optimal_labels
            n_clusters = len(set(optimal_labels)) - (1 if -1 in optimal_labels else 0)
            n_noise = sum(optimal_labels == -1)
            print(f"✅ Optimized clustering complete. Found {n_clusters} clusters and {n_noise} noise points.")
        else:
            print("🔄 Using fallback clustering...")
            # Fallback clustering
            if HDBSCAN_AVAILABLE:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=10,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
            else:
                clusterer = DBSCAN(eps=0.3, min_samples=5, metric='euclidean')

            self.cluster_labels = clusterer.fit_predict(self.reduced_embeddings)
            n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            n_noise = sum(self.cluster_labels == -1)
            algorithm_name = "HDBSCAN" if HDBSCAN_AVAILABLE else "DBSCAN"
            print(f"✅ {algorithm_name} clustering complete. Found {n_clusters} clusters and {n_noise} noise points.")

        return self.cluster_labels

    def find_optimal_clustering_parameters(self, reduced_embeddings):
        """Find optimal clustering parameters to avoid large clusters"""
        print("🔍 Finding optimal clustering parameters...")

        n_samples = len(reduced_embeddings)
        results = []

        if HDBSCAN_AVAILABLE:
            print("🔧 Testing HDBSCAN configurations...")

            # Focused parameter ranges for your dataset size
            min_cluster_sizes = [5, 8, 12, 15, 20, 25, 30, 40, 60]
            min_samples_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
            epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7]

            for min_cluster_size in min_cluster_sizes:
                for ratio in min_samples_ratios:
                    min_samples = max(1, int(min_cluster_size * ratio))
                    for epsilon in epsilons:
                        try:
                            clusterer = hdbscan.HDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_epsilon=epsilon,
                                metric='euclidean'
                            )

                            labels = clusterer.fit_predict(reduced_embeddings)

                            # Calculate metrics
                            unique_labels = np.unique(labels)
                            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                            n_noise = np.sum(labels == -1)
                            noise_ratio = n_noise / len(labels)

                            # Calculate quality score
                            if n_clusters > 0:
                                cluster_sizes = [np.sum(labels == i) for i in unique_labels if i != -1]
                                largest_cluster = max(cluster_sizes)
                                avg_cluster = np.mean(cluster_sizes)
                                largest_ratio = largest_cluster / len(labels)

                                # Quality scoring
                                silhouette = None
                                if n_clusters > 1 and noise_ratio < 0.9:
                                    try:
                                        non_noise_mask = labels != -1
                                        if np.sum(non_noise_mask) > 10:
                                            silhouette = silhouette_score(
                                                reduced_embeddings[non_noise_mask],
                                                labels[non_noise_mask]
                                            )
                                    except:
                                        silhouette = 0

                                # Scoring components
                                silhouette_component = silhouette * 30 if silhouette else 0

                                if 5 <= n_clusters <= 12:
                                    cluster_component = 20
                                elif 3 <= n_clusters <= 4 or 13 <= n_clusters <= 20:
                                    cluster_component = 15
                                else:
                                    cluster_component = -10

                                # Heavily penalize dominant clusters
                                if largest_ratio > 0.8:
                                    balance_component = -30
                                elif largest_ratio > 0.7:
                                    balance_component = -15
                                elif largest_ratio > 0.5:
                                    balance_component = -5
                                else:
                                    balance_component = 20

                                if noise_ratio <= 0.20:
                                    noise_component = 0
                                else:
                                    noise_component = -15

                                quality_score = silhouette_component + cluster_component + balance_component + noise_component
                            else:
                                quality_score = -1000
                                largest_cluster = 0
                                largest_ratio = 0

                            results.append({
                                'algorithm': 'HDBSCAN',
                                'config': {
                                    'min_cluster_size': min_cluster_size,
                                    'min_samples': min_samples,
                                    'cluster_selection_epsilon': epsilon
                                },
                                'labels': labels,
                                'n_clusters': n_clusters,
                                'noise_ratio': noise_ratio,
                                'largest_cluster_size': largest_cluster,
                                'largest_ratio': largest_ratio,
                                'quality_score': quality_score,
                                'silhouette': silhouette
                            })

                        except Exception as e:
                            continue

        else:
            print("🔧 Testing DBSCAN configurations...")
            eps_values = np.arange(0.1, 1.0, 0.05)
            min_samples_values = [3, 5, 8, 10, 15, 20, 30]

            for eps in eps_values:
                for min_samples in min_samples_values:
                    try:
                        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                        labels = clusterer.fit_predict(reduced_embeddings)

                        unique_labels = np.unique(labels)
                        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                        n_noise = np.sum(labels == -1)
                        noise_ratio = n_noise / len(labels)

                        # Simple quality scoring for DBSCAN
                        quality_score = n_clusters * 10 - noise_ratio * 50

                        if n_clusters > 0:
                            cluster_sizes = [np.sum(labels == i) for i in unique_labels if i != -1]
                            largest_cluster = max(cluster_sizes)
                            largest_ratio = largest_cluster / len(labels)
                        else:
                            largest_cluster = 0
                            largest_ratio = 0

                        results.append({
                            'algorithm': 'DBSCAN',
                            'config': {'eps': eps, 'min_samples': min_samples},
                            'labels': labels,
                            'n_clusters': n_clusters,
                            'noise_ratio': noise_ratio,
                            'largest_cluster_size': largest_cluster,
                            'largest_ratio': largest_ratio,
                            'quality_score': quality_score
                        })

                    except Exception as e:
                        continue

        # Find best configuration
        if not results:
            print("❌ No valid clustering configurations found!")
            return None, None

        best_result = max(results, key=lambda x: x['quality_score'])

        print(f"🏆 OPTIMAL CONFIGURATION:")
        print(f"   Clusters: {best_result['n_clusters']}")
        print(f"   Noise: {best_result['noise_ratio']:.1%}")
        print(f"   Largest cluster: {best_result['largest_ratio']:.1%}")

        return best_result['config'], best_result['labels']

    def save_sentences_with_clusters(self, filename="sentences_with_clusters_updated.xlsx"):
        """Save sentences with their cluster assignments to Excel"""
        print("🔹 Step 5: Saving sentences with cluster assignments...")

        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            print("⚠️  No cluster labels available. Skipping sentence saving.")
            return None

        # Add cluster labels to sentences dataframe
        sentences_with_clusters = self.sentences_df.copy()
        sentences_with_clusters['cluster'] = self.cluster_labels

        # Create a representative sentence for each cluster (for the avg_sentence column)
        cluster_avg_sentences = {}
        for cluster_id in set(self.cluster_labels):
            cluster_sentences = sentences_with_clusters[sentences_with_clusters['cluster'] == cluster_id]['sentence']
            if len(cluster_sentences) > 0:
                # Use the first sentence as a representative (you could use a more sophisticated method)
                cluster_avg_sentences[cluster_id] = cluster_sentences.iloc[0]
            else:
                cluster_avg_sentences[cluster_id] = ""

        # Add avg_sentence column
        sentences_with_clusters['avg_sentence'] = sentences_with_clusters['cluster'].map(cluster_avg_sentences)

        # Save to Excel
        sentences_with_clusters.to_excel(filename, index=False)
        print(f"Sentences with clusters saved to '{filename}'")

        return sentences_with_clusters

    def assemble_responses_with_cluster_chains(self, filename="responses_with_cluster_chains.xlsx"):
        """Assemble original responses with cluster chains"""
        print("🔹 Step 6: Assembling responses with cluster chains...")

        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            print("⚠️  No cluster labels available. Skipping response assembly.")
            return None

        # Add cluster labels to sentences dataframe
        sentences_with_clusters = self.sentences_df.copy()
        sentences_with_clusters['cluster'] = self.cluster_labels

        # Group by original response and create cluster chains
        response_groups = sentences_with_clusters.groupby(['original_row_index', 'original_response',
                                                           'Story', 'StudyNum', 'SubID'])

        assembled_responses = []

        for (orig_idx, orig_response, story, study_num, sub_id), group in response_groups:
            # Sort by sentence index to maintain order
            group_sorted = group.sort_values('sentence_index_in_response')

            # Create cluster chain
            cluster_chain = group_sorted['cluster'].tolist()
            cluster_chain_str = ' -> '.join(map(str, cluster_chain))

            # Count cluster occurrences
            cluster_counts = Counter(cluster_chain)

            assembled_responses.append({
                'original_row_index': orig_idx,
                'SubID': sub_id,
                'StudyNum': study_num,
                'Story': story,
                'original_response': orig_response,
                'num_sentences': len(group_sorted),
                'cluster_chain': cluster_chain_str,
                'cluster_chain_list': cluster_chain,
                'unique_clusters': len(set(cluster_chain)),
                'cluster_counts': dict(cluster_counts),
                'dominant_cluster': max(cluster_counts, key=cluster_counts.get) if cluster_counts else -1
            })

        self.responses_with_clusters = pd.DataFrame(assembled_responses)

        # Save to Excel
        self.responses_with_clusters.to_excel(filename, index=False)
        print(f"Responses with cluster chains saved to '{filename}'")

        return self.responses_with_clusters

    def visualize_clusters_with_strategy_names(self, figsize=(18, 12)):
        """Create comprehensive cluster visualizations with empathy strategy names"""
        print("🔹 Step 7: Creating visualizations with strategy names...")

        if self.reduced_embeddings is None or self.cluster_labels is None:
            print("⚠️  No clustering data available for visualization.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Main cluster plot with strategy names
        unique_labels = set(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters > 0:
            # Create color mapping
            palette = plt.cm.get_cmap('tab20', max(20, np.max(self.cluster_labels) + 1))
            colors = [palette(label % 20) if label >= 0 else (0.5, 0.5, 0.5, 0.5) for label in self.cluster_labels]
        else:
            colors = [(0.5, 0.5, 0.5, 0.5) for _ in self.cluster_labels]

        # Use t-SNE for better visualization
        try:
            print("Creating t-SNE visualization...")
            tsne_embeddings = self.create_tsne_visualization(
                self.reduced_embeddings,
                perplexity=min(30, len(self.reduced_embeddings) // 4),
                random_state=42
            )
            x_coords, y_coords = tsne_embeddings[:, 0], tsne_embeddings[:, 1]
            xlabel, ylabel = "t-SNE-1", "t-SNE-2"
        except Exception as e:
            print(f"⚠️  t-SNE failed, using UMAP first 2 components: {e}")
            x_coords, y_coords = self.reduced_embeddings[:, 0], self.reduced_embeddings[:, 1]
            xlabel, ylabel = "UMAP-1", "UMAP-2"

        # Plot clusters
        scatter = axes[0, 0].scatter(x_coords, y_coords, c=colors, s=10, alpha=0.7)
        algorithm_name = "HDBSCAN" if HDBSCAN_AVAILABLE else "DBSCAN"
        axes[0, 0].set_title(f"{algorithm_name} Clustering: Empathy Strategies\n{n_clusters} clusters identified")
        axes[0, 0].set_xlabel(xlabel)
        axes[0, 0].set_ylabel(ylabel)
        axes[0, 0].grid(True, alpha=0.3)

        # Create legend with strategy names
        legend_elements = []
        cluster_counts = Counter(self.cluster_labels)
        sorted_clusters = sorted([c for c in unique_labels if c != -1])

        for cluster_id in sorted_clusters:
            if cluster_id in EMPATHY_STRATEGY_NAMES:
                strategy_name = EMPATHY_STRATEGY_NAMES[cluster_id]
                count = cluster_counts[cluster_id]
                color = palette(cluster_id % 20) if cluster_id >= 0 else (0.5, 0.5, 0.5)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=color, markersize=8,
                                                  label=f"C{cluster_id}: {strategy_name} ({count})"))

        # Add noise to legend if present
        if -1 in unique_labels:
            noise_count = cluster_counts[-1]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=(0.5, 0.5, 0.5), markersize=8,
                                              label=f"Noise: General Sympathy ({noise_count})"))

        # Place legend outside the plot
        axes[0, 0].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
                          fontsize=9, title="Empathy Strategies")

        # 2. Cluster size distribution with strategy names
        non_noise_counts = {k: v for k, v in cluster_counts.items() if k != -1}

        if non_noise_counts:
            sorted_clusters_with_counts = sorted(non_noise_counts.items(), key=lambda x: x[1], reverse=True)
            clusters, counts = zip(*sorted_clusters_with_counts)

            bars = axes[0, 1].bar(range(len(clusters)), counts,
                                  color=[palette(c % 20) for c in clusters], alpha=0.8)
            axes[0, 1].set_title("Empathy Strategy Distribution")
            axes[0, 1].set_xlabel("Strategy Clusters")
            axes[0, 1].set_ylabel("Number of Sentences")

            # Create x-axis labels with strategy names
            strategy_labels = []
            for c in clusters:
                if c in EMPATHY_STRATEGY_NAMES:
                    name = EMPATHY_STRATEGY_NAMES[c]
                    # Truncate long names for readability
                    if len(name) > 15:
                        name = name[:15] + "..."
                    strategy_labels.append(f"C{c}\n{name}")
                else:
                    strategy_labels.append(f"C{c}")

            axes[0, 1].set_xticks(range(len(clusters)))
            axes[0, 1].set_xticklabels(strategy_labels, rotation=45, ha='right', fontsize=8)

            # Add noise information
            noise_count = cluster_counts.get(-1, 0)
            axes[0, 1].text(0.7, 0.9, f"Noise points: {noise_count}", transform=axes[0, 1].transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        else:
            axes[0, 1].text(0.5, 0.5, "No clusters found\n(all points are noise)",
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("Empathy Strategy Distribution")

        # 3. Response length vs unique clusters
        if self.responses_with_clusters is not None and len(self.responses_with_clusters) > 0:
            axes[1, 0].scatter(self.responses_with_clusters['num_sentences'],
                               self.responses_with_clusters['unique_clusters'],
                               alpha=0.6, color='coral')
            axes[1, 0].set_title("Response Length vs Empathy Strategy Diversity")
            axes[1, 0].set_xlabel("Number of Sentences in Response")
            axes[1, 0].set_ylabel("Number of Unique Strategies Used")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, "No response data\navailable",
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Response Length vs Strategy Diversity")

        # 4. Study vs strategy heatmap
        if self.responses_with_clusters is not None and len(self.responses_with_clusters) > 0:
            try:
                study_cluster_data = []
                for _, row in self.responses_with_clusters.iterrows():
                    for cluster in row['cluster_chain_list']:
                        if cluster != -1:
                            study_cluster_data.append({
                                'StudyNum': row['StudyNum'],
                                'Cluster': cluster,
                                'Strategy': EMPATHY_STRATEGY_NAMES.get(cluster, f"Cluster {cluster}")
                            })

                if study_cluster_data:
                    study_cluster_df = pd.DataFrame(study_cluster_data)
                    pivot_table = study_cluster_df.pivot_table(
                        index='StudyNum', columns='Cluster', aggfunc=len, fill_value=0
                    )

                    # Limit to top clusters for readability
                    if pivot_table.shape[1] > 12:
                        top_clusters = pivot_table.sum().nlargest(12).index
                        pivot_table = pivot_table[top_clusters]

                    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
                    axes[1, 1].set_title("Study vs Empathy Strategy Usage")
                    axes[1, 1].set_xlabel("Strategy Cluster")
                    axes[1, 1].set_ylabel("Study Number")

                    # Add strategy names to x-axis
                    cluster_names = []
                    for col in pivot_table.columns:
                        if col in EMPATHY_STRATEGY_NAMES:
                            name = EMPATHY_STRATEGY_NAMES[col]
                            if len(name) > 12:
                                name = name[:12] + "..."
                            cluster_names.append(f"C{col}\n{name}")
                        else:
                            cluster_names.append(f"C{col}")

                    axes[1, 1].set_xticklabels(cluster_names, rotation=45, ha='right', fontsize=8)
                else:
                    axes[1, 1].text(0.5, 0.5, "No cluster data\navailable for heatmap",
                                    ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title("Study vs Empathy Strategy Usage")
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f"Heatmap error:\n{str(e)[:50]}...",
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title("Study vs Empathy Strategy Usage")
        else:
            axes[1, 1].text(0.5, 0.5, "No response data\navailable for heatmap",
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Study vs Empathy Strategy Usage")

        plt.tight_layout()
        plt.savefig('empathy_strategy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_cluster_characteristics_with_names(self):
        """Analyze and display cluster characteristics with strategy names"""
        print("🔹 Step 8: Analyzing empathy strategy characteristics...")

        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            print("⚠️  No cluster labels available for analysis.")
            return

        sentences_with_clusters = self.sentences_df.copy()
        sentences_with_clusters['cluster'] = self.cluster_labels

        print("\n" + "=" * 60)
        print("EMPATHY STRATEGY ANALYSIS SUMMARY")
        print("=" * 60)

        for cluster_id in sorted(set(self.cluster_labels)):
            strategy_name = EMPATHY_STRATEGY_NAMES.get(cluster_id, f"Unknown Strategy")

            if cluster_id == -1:
                print(f"\n🔸 NOISE POINTS - {strategy_name}:")
            else:
                print(f"\n🔸 CLUSTER {cluster_id} - {strategy_name}:")

            cluster_sentences = sentences_with_clusters[sentences_with_clusters['cluster'] == cluster_id]
            print(f"   📊 Size: {len(cluster_sentences)} sentences")

            # Show sample sentences
            sample_sentences = cluster_sentences['sentence'].head(3).tolist()
            print(f"   📝 Sample sentences:")
            for i, sent in enumerate(sample_sentences, 1):
                print(f"      {i}. {sent[:80]}...")

            # Study distribution
            study_dist = cluster_sentences['StudyNum'].value_counts()
            print(f"   📚 Study distribution: {dict(study_dist)}")


def choose_masking_strategy():
    """Interactive masking strategy selection"""
    print("\n🎯 MASKING STRATEGY OPTIONS:")
    print("=" * 50)
    print("1. none      - No masking (keep original sentences)")
    print("2. entities  - Mask names, places, organizations → [PERSON], [LOCATION], [ORG]")
    print("3. pronouns  - Normalize pronouns → [I], [YOU], [HE/SHE], [THEY], [WE]")
    print("4. combined  - Entities + Pronouns (🌟 RECOMMENDED)")
    print("5. subjects  - Mask sentence subjects → [SUBJECT]")
    print("6. full      - All strategies combined (most aggressive)")

    while True:
        choice = input(f"\nSelect strategy (1-6) [default: 4 - combined]: ").strip()

        masking_map = {
            '1': 'none', '2': 'entities', '3': 'pronouns',
            '4': 'combined', '5': 'subjects', '6': 'full', '': 'combined'
        }

        if choice in masking_map:
            strategy = masking_map[choice]
            print(f"\n✅ Selected: {strategy}")
            return strategy
        else:
            print("❌ Invalid choice. Please enter 1-6.")


def main():
    # Configuration - YOUR FIXED PARAMETERS
    FIXED_N_COMPONENTS = 15
    FIXED_N_NEIGHBORS = 3
    FIXED_MIN_DIST = 0.001

    csv_file = 'StoryDBfull_24012025_Public.csv'

    print("🔬 STREAMLINED EMPATHY STRATEGY CLUSTERING")
    print("=" * 60)
    print("Using your optimized configuration:")
    print(f"   📊 Dimensions: {FIXED_N_COMPONENTS}")
    print(f"   👥 Neighbors: {FIXED_N_NEIGHBORS}")
    print(f"   📏 Min distance: {FIXED_MIN_DIST}")

    # Interactive masking strategy selection
    masking_strategy = choose_masking_strategy()

    # Interactive study selection
    study_num = ['1', '1a', '1b', '1c', '1d', '2b', '2a', '3', '4', '5']
    chosen_studies = []

    print(f"\n📚 STUDY SELECTION")
    print("=" * 30)
    print("Available studies:", ', '.join(study_num))

    while True:
        unchosen_study = [item for item in study_num if item not in chosen_studies]
        if not unchosen_study:
            break

        print(f"\nRemaining studies: {unchosen_study}")
        print(f"Currently selected: {chosen_studies if chosen_studies else 'None'}")

        study = input(
            "\nOptions:\n"
            "• Enter study number to add (e.g., '1', '2a')\n"
            "• Type 'all' to select all remaining studies\n"
            "• Type 'done' to proceed with current selection\n"
            "• Type 'clear' to start over\n"
            "Choice: "
        ).strip()

        if study == 'all':
            chosen_studies.extend(unchosen_study)
            print(f"✅ Added all remaining studies: {unchosen_study}")
        elif study == 'done':
            break
        elif study == 'clear':
            chosen_studies = []
            print("🔄 Cleared selection")
        elif study in unchosen_study:
            chosen_studies.append(study)
            print(f"✅ Added study {study}")
        elif study in chosen_studies:
            print(f"⚠️  Study {study} already selected")
        else:
            print(f"❌ Invalid study '{study}'. Please try again.")

    if not chosen_studies:
        print("❌ No studies selected. Exiting.")
        return

    print(f"\n🎯 Final selection: {chosen_studies}")
    confirm = input("Proceed with these studies? (y/n) [y]: ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("Exiting...")
        return

    # Initialize analyzer
    analyzer = SentenceClusteringAnalyzer(masking_strategy=masking_strategy)
    start_time = time.time()

    try:
        # Run streamlined pipeline
        print(f"\n🚀 RUNNING STREAMLINED ANALYSIS")
        print("=" * 40)

        analyzer.load_and_prepare_data(csv_file, chosen_studies)
        analyzer.create_embeddings()

        # Use your fixed configuration
        analyzer.run_fixed_configuration(
            n_components=FIXED_N_COMPONENTS,
            n_neighbors=FIXED_N_NEIGHBORS,
            min_dist=FIXED_MIN_DIST
        )

        # Save results
        analyzer.save_sentences_with_clusters()
        analyzer.assemble_responses_with_cluster_chains()

        # Create visualizations with strategy names
        analyzer.visualize_clusters_with_strategy_names()

        # Analyze characteristics with strategy names
        analyzer.analyze_cluster_characteristics_with_names()

        # Success summary
        print(f"\n🎉 ANALYSIS COMPLETED!")
        print("=" * 50)
        print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
        print(f"🎭 Masking strategy: {masking_strategy}")
        print(f"📊 Configuration: {FIXED_N_COMPONENTS}D, {FIXED_N_NEIGHBORS} neighbors, {FIXED_MIN_DIST} min_dist")
        print(f"📝 Total sentences: {len(analyzer.sentences_df)}")
        print(f"🎯 Clusters found: {len(set(analyzer.cluster_labels)) - (1 if -1 in analyzer.cluster_labels else 0)}")

        print(f"\n📁 FILES CREATED:")
        print("   • sentences_with_clusters_updated.xlsx - Sentences with empathy strategy labels")
        print("   • responses_with_cluster_chains.xlsx - Responses with strategy sequences")
        print("   • empathy_strategy_analysis.png - Visualization with strategy names")
        print("   • embeddings.npy - Sentence embeddings")

        print(f"\n🎯 EMPATHY STRATEGIES IDENTIFIED:")
        cluster_counts = Counter(analyzer.cluster_labels)
        for cluster_id in sorted(set(analyzer.cluster_labels)):
            if cluster_id in EMPATHY_STRATEGY_NAMES:
                strategy_name = EMPATHY_STRATEGY_NAMES[cluster_id]
                count = cluster_counts[cluster_id]
                print(f"   • Cluster {cluster_id}: {strategy_name} ({count} sentences)")

    except FileNotFoundError:
        print(f"❌ ERROR: Could not find data file '{csv_file}'")
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")

    print(f"\n👋 Empathy strategy analysis complete!")


if __name__ == "__main__":
    main()