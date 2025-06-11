# Huji-research-empathy_strategies
this work is a self independent research being conducted in the empathy lab of the Hebrew university of jersusalem.
it was done by the supervision of phd student Matan rubin and PI Pr.Anat Perry.

in this research we ask what empaty strategies are used by AI in different scenarios.
to answer this question we run a clustering algorithm on the sentence embedding from the AI responses.
in this repository you'll find 3 scripts: main.py,cluster_ratings_analysis.py and cluster_avg_sentence.py
a brief overview of what these file does is detailed below:

main.py - this script does the main analyzing of the research. it seperates the AI generated responses into individual sentences,
than encoding the sentences into the embedding space using BERT transforner.for the clustering, do dimentionallity reduction to 15 using UMAP algorithm.
for the clustering itself i used hdbscan algorithm.

cluster_ratings_analysis.py - this script takes the clusters given from the main.py script and run 'ANOVA' statistical analysis vs the empathy ratings from the users.
it eventuallyu produce a visuallization of the results.

cluster_avg_sentence.py - this script calculates the avg sentence in each cluster by encoding the sentences to the embsdding space (again using BERT). calculate the mean vector
of each cluster and decode it back into a sentence.
