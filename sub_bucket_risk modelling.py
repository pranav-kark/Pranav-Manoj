import pandas as pd
import numpy as np
from gensim.models import Word2Vec


class sub_buckets:
    def __init__(self, file):
        self.data = file

        embeddings = self.create_word_embeddings(self.data, 'Keywords')
        self.iterate_legends(embeddings)

    def create_word_embeddings(self, data, column):
        word_embeddings = {}
        sentences = []

        # Preprocess sentences and skip non-string values
        for value in data[column].values:
            if isinstance(value, str):
                sentences.append(value.split())

        # Train Word2Vec model
        model = Word2Vec(sentences, vector_size=100, min_count=1, window=5, workers=4)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        # Store word embeddings in a dictionary
        for word in model.wv.index_to_key:
            word_embeddings[word] = model.wv.get_vector(word)

        return word_embeddings

    def iterate_legends(self, embeddings):
        count = 0

        self.data['Keywords'] = self.data['Keywords'].astype(str).str.lower().replace('nan', '')
        self.data['Legend'] = self.data['Legend'].str.strip()

        # Check if 'cluster' column exists, create it if not
        if 'cluster' not in self.data.columns:
            self.data['cluster'] = ''
            
        for index, value in enumerate(self.data['Legend']):
            if value == 'Legal Risk':
                self.legal(index, embeddings)
            elif value == 'Security Risk':
                self.security(index, embeddings)
            elif value == 'Natural Risk':
                self.natural(index, embeddings)
            elif value == 'Reputational Risk':
                self.reputational(index, embeddings)
            elif value == 'Environmental Risk':
                self.environmental(index, embeddings)
            elif value == 'Operational Risk':
                self.operational(index, embeddings)
            elif value == 'Medical Risk':
                self.medical(index, embeddings)
            elif value == 'Political Risk':
                self.political(index, embeddings)
            elif value == 'Economic Risk':
                self.economic(index, embeddings)
            elif value == 'Geopolitical Risk':
                self.geopolitical(index, embeddings)
            elif value == 'Emerging Risk':
                self.emerging(index, embeddings)            
            else:
                self.data.at[index, 'cluster'] = value

        self.data['cluster'] = self.data.apply(
            lambda row: row['Legend'] if pd.isnull(row['cluster']) or row['cluster'] == [] or (
                        isinstance(row['cluster'], str) and row['cluster'].strip() == '') else row['cluster'], axis=1)

        return self.data

    def political(self, index, embeddings):
        sub_buckets = {
    'Protests and Activism': ['protests', 'antigovernment', 'mass', 'movement', 'unrest'],
    'Elections and Political Processes': ['political', 'election', 'electoral', 'fraud'],
    'Geopolitical Tensions': ['geopolitical', 'tensions', 'breach', 'trust', 'security'],
    'Regime and Governance': ['regime', 'change', 'constitutional', 'government', 'governance'],
    'Political Violence and Terrorism': ['violence', 'military', 'terrorist', 'terrorism'],
    'Economic and Regulatory Impact': ['regulatory', 'instability', 'nationalization', 'corruption', 'industry']
}
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:
                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def economic(self, index, embeddings):
        sub_buckets = {
    'Macroeconomic Indicators': ['economy', 'inflation', 'recession', 'monetary', 'policy'],
    'Debt and Financial Stability': ['debt', 'sanctions', 'banking', 'risk', 'crisis'],
    'International Economy': ['international', 'global', 'trade', 'sector', 'world'],
    'Unemployment and Labor Market': ['unemployment', 'employment', 'job', 'labor'],
    'Government Revenue and Fiscal Policy': ['revenue', 'fund', 'tax', 'fiscal', 'deficit']
}
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:

                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def legal(self, index, embeddings):
        sub_buckets = {
    'Sentencing and Prisons': ['sentence', 'prison'],
    'Lawsuits and Legal Proceedings': ['lawsuit', 'filed', 'contempt', 'court'],
    'Legislation and Constitutional Law': ['law', 'bill', 'constitutional', 'amendment', 'legislative'],
    'Regulatory Compliance and Scrutiny': ['noncompliance', 'regulatory', 'scrutiny'],
    'Legal Disputes and Controversies': ['dispute', 'legal', 'intellectual', 'property', 'disputes'],
    'Protests and Mass Action': ['protests', 'mass'],
    'Criminal Offenses and Investigations': ['murder', 'killing', 'assault', 'harassment', 'rape', 'kidnapping'],
    'Energy and Corporate Law': ['energy', 'chevron', 'corporation'],
    'Political Interference and Terrorism': ['interference', 'terrorism']
}
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:

                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def operational(self, index, embeddings):
        sub_buckets = {
'Breach and Data Security': ['breach', 'dos', 'company', 'llp', 'firm', 'data'],
'Cybercrime and Attacks': ['cybercrime', 'attack', 'security', 'malware', 'phishing'],
'Regulatory Compliance and Risk': ['regulatory', 'control', 'industry', 'risks'],
'Legal and Lawsuits': ['lawsuit', 'filed', 'law', 'constitutional', 'lawsuit'],
'Disruptions and Infrastructure': ['disruptions', 'infrastructure', 'disaster', 'plant'],
'Corporate Governance': ['governance', 'corporate', 'property'],
'Environmental Impact and Pollution': ['moment', 'pollution', 'warming']
}
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:

                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def natural(self, index, embeddings):
        sub_buckets = {
    'Weather': ['weather', 'storm', 'cyclone', 'fog', 'heatwave', 'rainfall', 'heavy', 'waves', 'hurricane'],
    'Earthquake': ['earthquake', 'earthquakes'],
    'Flood': ['flooding', 'flood', 'floods'],
    'Drought': ['drought', 'droughts'],
    'Landslide': ['landslides'],
    'Wildfire': ['wildfires', 'wildfire'],
    'Tornado': ['tornado', 'tornadoes'],
    'Climate Change': ['change', 'climate', 'environment', 'warming'],
    'Disaster': ['disaster', 'calamity'],
    'Power Outages': ['power', 'outages'],
    'Air Pollution': ['pollution'],
    'Natural Hazards': ['new', 'events', 'habitat']
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:
                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def security(self, index, embeddings):
        sub_buckets = {
    'Violent Crime': ['murder', 'killing', 'assault', 'shooting', 'rape', 'robbery'],
    'Crime Investigation': ['arrest', 'bomb', 'conspiracy', 'kidnapping', 'fraud'],
    'Threat and Security': ['threat', 'harassment', 'warning', 'human', 'trafficking', 'gunman'],
    'Property Crime': ['burglary', 'firing', 'extortion', 'arson', 'abduction'],
    'Public Disorder': ['riot', 'scare', 'breach', 'trust', 'evacuation', 'shootout']
}
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:
                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def reputational(self, index, embeddings):
        sub_buckets = {
    'Company': ['company', 'firm', 'llc', 'llp', 'plc'],
    'Fraud': ['fraud', 'scandals', 'lawsuit', 'filed', 'theft'],
    'Information Security': ['systems', 'software', 'devices', 'intellectual', 'property', 'cybercrime', 'malware', 'ransomware', 'viruses', 'data', 'hardware'],
    'Industry and Infrastructure': ['industry', 'protection', 'infrastructure', 'critical', 'power', 'security', 'banking', 'manufacturing', 'pharmaceutical', 'refinery', 'mining', 'oil', 'energy', 'renewable', 'pollution', 'trade', 'regulatory', 'noncompliance', 'steel'],
    'Political and Legal': ['protests', 'governance', 'sector', 'access', 'control', 'trademarks', 'fake', 'news', 'activism', 'breach', 'new', 'law', 'mitigation', 'asset', 'trade', 'rape', 'assault', 'noncompliance', 'foreign', 'aviation', 'bill', 'military'],
    'Healthcare': ['medical', 'healthcare'],
    'Terrorism and Conflict': ['taliban', 'afghanistan']
}

        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:

                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def medical(self, index, embeddings):
        sub_buckets = {
    'Disease Outbreak': ['disease', 'communicable', 'infection', 'viral', 'epidemic', 'outbreak'],
    'Medical Malpractice': ['malpractice', 'misdiagnosis', 'delayed', 'diagnosis', 'injured', 'patient'],
    'Fire and Safety': ['fire', 'factory', 'blast', 'accident', 'emergency', 'incident'],
    'Workplace Health': ['workplace', 'worker', 'lawsuit', 'pollution', 'healthcare', 'employee'],
    'Terrorism and Security': ['terrorist', 'security', 'terrorism', 'harassment', 'extinction', 'rebellion'],
    'Natural Disasters': ['weather', 'storm', 'tornado', 'flood', 'earthquake', 'flooding'],
    'Pharmaceutical Industry': ['medical', 'industry', 'pharmaceutical', 'devices', 'bp', 'shell']
}

        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:

                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def environmental(self, index, embeddings):
        sub_buckets = {
    'Pollution': ['pollution', 'air', 'water', 'scarcity', 'waste', 'hazardous'],
    'Climate Change': ['extinction', 'rebellion', 'global', 'warming', 'change', 'climate'],
    'Natural Disasters': ['fire', 'forest', 'wildfires', 'flood', 'storm', 'drought'],
    'Medical Risks': ['misdiagnosis', 'treatment', 'disease', 'communicable', 'pharmaceutical', 'virus'],
    'Infrastructure Risks': ['power', 'railways', 'refinery', 'infrastructure', 'mining', 'manufacturing'],
    'Environmental Regulations': ['protection', 'industry', 'sector', 'governance', 'property', 'regulatory'],
    'Security Risks': ['security', 'terrorism', 'bomb', 'violence', 'murder', 'crime']
}
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:

                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def geopolitical(self, index, embeddings):
        sub_buckets = {
    'War': ['war', 'ukraine', 'russia', 'conflict', 'pakistan', 'china', 'india', 'afghanistan', 'taliban'],
    'Crisis': ['crisis', 'security', 'economic', 'threat', 'risks', 'terrorism', 'power', 'protection'],
    'Infrastructure': ['critical', 'infrastructure', 'plant', 'refinery', 'railways', 'sector'],
    'Natural Disasters': ['earthquake', 'storm', 'weather'],
    'International Relations': ['national', 'treaty', 'alliance', 'sovereignty', 'germany', 'iran', 'poland', 'japan'],
    'Political Unrest': ['scare', 'protests', 'rape', 'political', 'argentina', 'iraq', 'tensions', 'change'],
    'Geopolitics': ['geopolitical', 'israel', 'france', 'australia', 'sweden', 'spain', 'belarus', 'north'],
    'Crime': ['crime', 'killing', 'rape', 'assault', 'murder', 'robbery', 'fraud', 'shooting'],
    'Human Rights': ['human', 'rights', 'violations', 'sanctions', 'immigrant']
}
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:
                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

    def emerging(self, index, embeddings):
        sub_buckets = {
    'Flooding': ['flooding', 'flood', 'drought', 'storm', 'water', 'rainfall'],
    'Labor Disputes': ['strike', 'union', 'roadblock', 'agitation', 'dharna'],
    'Power Outages': ['power', 'outages', 'electricity', 'outage'],
    'Public Protests': ['riot', 'protests', 'demonstration', 'outrage', 'unrest'],
    'Infrastructure Failure': ['building', 'collapse', 'bridge', 'refinery', 'section'],
    'Environmental Hazard': ['pollution', 'climate', 'environment', 'atmosphere', 'disaster'],
    'Terrorism': ['terrorism', 'terrorist', 'attack', 'bomb', 'explosive'],
    'Security Threats': ['security', 'threat', 'violence', 'assault', 'shooting'],
    'Transportation Issues': ['railways', 'train', 'traffic', 'road', 'curfew'],
    'Political Unrest': ['political', 'government', 'election', 'unrest', 'instability'],
    'Health Crisis': ['disease', 'pandemic', 'outbreak', 'medical', 'communicable'],
    'Cybersecurity': ['malware', 'hacker', 'phishing', 'data', 'breach'],
    'Business Disruption': ['industry', 'business', 'economy', 'company', 'disruption'],
    'Social Issues': ['lgbtq', 'caste', 'refugee', 'migrant', 'citizenship'],
    'Crime': ['crime', 'murder', 'rape', 'kidnapping', 'robbery']
}
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in sub_buckets.items():
                if any(keyword in words for keyword in keywords):
                    assigned_clusters.append(cluster)
                else:
                    cluster_vecs = [embeddings.get(word, np.zeros(100)) for word in words]
                    keyword_vecs = [embeddings.get(keyword, np.zeros(100)) for keyword in keywords]
                    cluster_vec = np.mean(cluster_vecs, axis=0)
                    keyword_vec = np.mean(keyword_vecs, axis=0)

                    cluster_vec = cluster_vec.reshape(1, -1)
                    keyword_vec = keyword_vec.reshape(1, -1)

                    similarity = np.dot(cluster_vec, keyword_vec.T) / (
                                np.linalg.norm(cluster_vec) * np.linalg.norm(keyword_vec))
                    similarity = similarity[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster
            if most_similar_cluster is not None:
                assigned_clusters.append(most_similar_cluster)
            assigned_clusters_str = ', '.join(assigned_clusters) if assigned_clusters else None
            self.data.at[index, 'Sub Bucket'] = assigned_clusters_str

        return self.data

df = pd.read_csv('Combined Risk Dataset.csv')
output=sub_buckets(df)
updated_data = output.data
updated_data.to_csv('Output.csv')
Final_Data_Frame=pd.DataFrame(updated_data)