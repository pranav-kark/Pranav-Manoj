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
        for index, value in enumerate(self.data['Legend']):
            if value == 'Security':
                self.security(index, embeddings)
            elif value == 'Social':
                self.social(index, embeddings)
            elif value == 'Environment':
                self.enviroment(index, embeddings)
            elif value == 'Education':
                self.education(index, embeddings)
            elif value == 'Economy':
                self.economy(index, embeddings)
            elif value == 'Political':
                self.political(index, embeddings)
            elif value == 'Crime':
                self.crime(index, embeddings)
            elif value == 'Health':
                self.health(index, embeddings)
            elif value == 'Terrorism':
                self.terrorism(index, embeddings)
            elif value == 'Technology':
                self.technology(index, embeddings)
            elif value == 'Military':
                self.military(index, embeddings)
            elif value == 'Legal':
                self.legal(index, embeddings)
            elif value == 'Disaster':
                self.disaster(index, embeddings)
            else:
                self.data.at[index, 'cluster'] = value

        self.data['cluster'] = self.data.apply(
            lambda row: row['Legend'] if pd.isnull(row['cluster']) or row['cluster'] == [] or (
                        isinstance(row['cluster'], str) and row['cluster'].strip() == '') else row['cluster'], axis=1)

    def legal(self, index, embeddings):
        clusters = {
            'Court-related': ['court', 'supreme', 'jurisdiction', 'tribunal', 'judicial', 'judgment', 'review',
                              'contempt', 'case', 'pending'],
            'Other relevant topics': ['new', 'policy', 'approval', 'labor', 'sedition', 'election', 'regime', 'moto',
                                      'icj', 'lok', 'adalat', 'procedure'],
            'Legal system and Laws': ['legal', 'law', 'constitutional', 'amendment', 'federal', 'international',
                                      'civil', 'rights', 'solicitor', 'intellectual', 'property', 'regulation'],
            'Litigation and legal processings': ['litigation', 'order', 'lawsuit', 'filed', 'justice', 'procedure'],
            'Criminal law and Justice': ['criminal', 'bail', 'attorney', 'prison', 'sentence'],
            'Discrimination and Human Rights': ['discrimination', 'general', 'interest', 'fundamental', 'privacy',
                                                'rights'],
            'Trademarks and Copyrights': ['trademark', 'copyrights'],
            'Family and Social Issues': ['family', 'abortion'],
            'Legal institutions and Bodies': ['high', 'public', 'chief', 'India', 'icj', 'lok', 'adalat']
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def military(self, index, embeddings):
        clusters = {
            'Military Operations': ['military', 'army', 'navy', 'soldiers', 'force', 'war', 'attack'],
            'Defence and Security': ['security', 'troops', 'defence',
                                     'terrorists', 'missile', 'aircraft',
                                     'shot', 'fire'],
            'Border Control': ['border', 'coast', 'base', 'front', 'territory'],
            'International Alliance': ['nato', 'alliance', 'coalition', 'neutral', 'pact', 'norad', 'un',
                                       'council'],
            'Logistics and Support': ['service', 'group', 'company', 'intelligence', 'armed', 'forces',
                                      'equipment', 'operation']
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def technology(self, index, embeddings):
        clusters = {

            'Technology': ['technology', 'ai', 'media', 'innovation', 'information', 'design', 'electronics',
                           'computing', 'algorithms', 'cloud', 'internet', 'things', 'network',
                           'nanotechnology', 'computer'],
            'Healthcare': ['medical', 'healthcare', 'pharmaceuticals', 'devices', 'hospitality'],
            'Transportation': ['aviation', 'logistics', 'roads', 'shipping',
                               'marine', 'railways', 'automobile'],
            'Energy': ['power', 'electricity', 'renewable', 'energy', 'gas'],
            'Science': ['science', 'intelligence', 'robotics', 'aerospace', 'chemical', 'nanotechnology', 'metal',
                        'genome', 'sequencing', 'scientific']
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def terrorism(self, index, embeddings):
        clusters = {
            'Afghanistan & Pakistan': ['afghanistan', 'taliban', 'attack', 'pakistan', 'kashmir', 'jammu', 'poonch', 'rajouri'],
            'Internation Terrorism': ['terrorism', 'terrorist', 'terroristattack', 'isis', 'islamic', 'state',
                                      'activity', 'gun', 'explosive', 'bomb', 'suicide', 'militant', 'jihadist'],
            'Global Security': ['united', 'states', 'russia', 'iran', 'china', 'germany', 'qatar', 'ukraine', 'israel',
                                'syria', 'spain', 'france', 'belgium', 'switzerland', 'arab', 'emirates', 'naxal',
                                'chhattisgarh', 'punjab'],
            'Counter-Terrorism Measures': ['security', 'funding', 'act', 'group', 'defence', 'surveillance',
                                           'intelligence'],
            'Regional Conflits': ['india', 'iraq', 'turkey', 'saudi', 'arabia', 'nigeria', 'somalia', 'libya', 'mali',
                                  'yemen', 'south', 'sudan', 'congo', 'burkina', 'faso', 'niger', 'chad'],
            'Terrorism Incidents': ['explosion', 'shooting', 'gunshot', 'arson', 'massacre', 'poisoning', 'pipe',
                                    'firearms', 'pistol', 'shotgun', 'knives', 'sword', 'rocket', 'missile']
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def health(self, index, embeddings):
        clusters = {
            'General Health': ['health', 'medical', 'emergency', 'treatment', 'research', 'community', 'medicine',
                               'hospital', 'patient', 'national', 'institutes', 'ministry', 'family', 'welfare',
                               'rural', 'women'],
            'Infectious Disease': ['covid', 'disease', 'pandemic', 'vaccine', 'virus', 'outbreaks', 'infection',
                                   'outbreak', 'flu', 'measles', 'cholera', 'epidemic', 'polio', 'malaria',
                                   'tuberculosis', 'aids', 'monkeypox', 'ebola', 'child', 'immune', 'system'],
            'Pharmaceuticals': ['drug', 'pharmaceutical'],
            'Biotechnology': ['artificial', 'biotechnology', 'genome', 'editing', 'mutations', 'pathogen', 'virology',
                              'assembly', 'sequencing'],
            'Health Organizations': ['world', 'organization', 'association', 'mission', 'maya'],
            'Specific Diseases': ['syndrome', 'bacteria', 'cure', 'intoxication', 'biotherapeutics'],
            'Health Programs': ['programme', 'ayush', 'one', 'welfare', 'rural', 'women']
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def crime(self, index, embeddings):
        clusters = {
            'Violent Crime': ['murder', 'killing', 'assault', 'shooting', 'rape', 'robbery', 'crime'],
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

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def disaster(self, index, embeddings):
        clusters = {
            'Natural Disaster': [
                'disaster,earthquake', 'earthquake,disaster', 'disaster,flooding', 'disaster,flood',
                'disaster,earthquake,earthquake', 'disaster,disaster,earthquake', 'disaster,disaster,landslides',
                'disaster,flooding,flood', 'disaster,flooding,heavy', 'earthquake,disaster,earthquake',
                'flood,disaster', 'disaster,flooding,flood', 'disaster,flooding,heavy',
                'earthquake,disaster,earthquake', 'disaster', 'disaster,storm', 'disaster,cyclone', 'disaster,accident',
                'disaster,cyclone,cyclone', 'storm,disaster', 'accident,disaster', 'cyclone,disaster',
                'disaster,heavy', 'disaster,storm,disaster', 'disaster,disaster,storm', 'disaster,cyclone,storm',
                'accident', 'rainfall', 'disaster,landslides,flooding', 'disaster,drought', 'disaster,flash', 'earthquake'
            ],
            'Nuclear Incident': [
                'nuclear', 'disaster,nuclear'
            ],
            'Hazardous Material Incident': [
                'leak', 'disaster,gas', 'disaster,disaster,gas', 'toxic'
            ],
            'Human Casualties': [
                'injured,disaster', 'disaster,injured', 'disaster,injuries', 'injured,disaster,fatalities',
                'injured,disaster,power', 'injured,disaster,landslides,heavy',
                'injuries,earthquake,earthquake,flooding,disaster,disaster,injuries,evacuations,gas', 'injured'
            ],
            'Structural Collapse': [
                'collapse', 'disaster,building'
            ],
            'Evacuation': [
                'evacuations,disaster'
            ],

            'Wildfire Event': ['disaster,heatwave', 'disaster,wildfire', 'wildfire,disaster', 'wildfires,disaster',
                               'disaster,wildfires'],
            'Manmade Disaster': ['train,train,accident,disaster', 'disaster,accident', 'leak,gas', 'accident',
                                 'nuclear', 'disaster,accident', 'leak']
        }

        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def enviroment(self, index, embeddings):

        clusters = {

            'International Climate Conference': ['cop27', 'ｃｏｐ２７', 'unfccc'],

            ' Climate Change and Pollution': ['climate', 'change', 'pollution', 'air', 'warming', 'change,environment',
                                              'change,climate'],

            'Biodiversity and Wildlife': ['wildlife', 'extinction', 'fauna'],

            'Natural Resources and Environment Quality': ['quality', 'environment', 'resources', 'natural',
                                                          'degradation', 'sanitation', 'waste', 'environment,climate',
                                                          'quality,air', 'biodiversity', 'global', 'food', 'security',
                                                          'flora', 'environmental', 'water', 'environment,science',
                                                          'pollution,air', 'hazards', 'pollution,pollution', 'save',
                                                          'science,environment', 'environmental,sustainability', 'go',
                                                          'green', 'issues', 'forest', 'environmental,disaster',
                                                          'environmental,climate', 'fire',
                                                          'sustainability,environment'],

            'Rebellion and Environmental Crisis': ['rebellion', 'crisis', 'spill'],

            'Mining and Oil': ['mining', 'management', 'oil', 'spill']
        }

        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def social(self, index, embeddings):
        clusters = {
            'Trade': [
                'trade'
            ],
            'Strike and Committee': [
                'strike,committee',
                'committee,strike',
                'union,strike',
                'committee,movement',
                'movement,strike,committee',
                'committee,riot',
                'committee,roadblocks',
                'strike,citizenship',
                'committee,social',
                'riot,committee',
                'strike,farmers',
                'heritage,committee'
            ],
            'Union and Movement': [
                'union,movement',
                'immigrant,movement',
                'movement,union',
                'reservation,movement',
                'movement,committee',
                'union',
                'movement,immigrant',
                'rights,movement',
                'justice,movement',
                'movement,indigenous',
                'migrant,public'
            ],
            'Social and Justice': [
                'social',
                'justice',
                'justice,committee',
                'justice,lgbtq',
                'justice,reservation'
            ],
            'LGBTQ+ Movement': [
                'lgbtq,movement',
                'movement,lgbtq',
                'lgbtq,social'
            ],
            'Heritage, Cultural and Festival': [
                'heritage',
                'cultural',
                'festival,cultural',
                'movement,festival',
                'festival,movement',
                'heritage,festival',
                'festival,strike',
                'temple,cultural',
                'communal',
                'strike,festival',
                'heritage,temple'
            ],
            'Farmer and Agitation': [
                'farmer,movement',
                'farmers,movement',
                'farmers,agitation',
                'farmers,strike'
            ],
            'Caste and Social': [
                'movement,caste',
                'caste,social',
                'caste,movement',
                'reservation,movement',
                'caste,caste,social'
            ],
            'Temple and Strike': [
                'temple,strike',
                'strike,temple',
                'temple,movement',
                'temple,cultural'
            ],
            'Tribe and Riot': [
                'tribe,riot',
                'tribe,cultural'
            ],
            'Public and Rights': [
                'public',
                'rights'
            ],
            'Emergency': [
                'emergency',
                'emergency,migrant'
            ],
            'Refugee Movement': [
                'movement,refugee',
                'refugee,movement'
            ],
            'Reservation Movement': [
                'strike,reservation',
                'reservation,movement'
            ],
            'National': [
                'national'
            ]
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def education(self, index, embeddings):

        clusters = {
            'University Education': ['university,education', 'secondary', 'education,university',
                                     'university,education', 'university,development,education',
                                     'university,university,education', 'university,literacy'],
            'Development and Policy': ['development,education', 'development,development,education',
                                       'education,development', 'development,university,education',
                                       'university,development,education', 'education,development,education',
                                       'development'],
            'Issues and Challenges': ['issues,education', 'education,education', 'education,issues', 'education,policy',
                                      'policy,education', 'education,challenge', 'education,international',
                                      'education,problems', 'policy'],
            'Education Initiatives and Development': ['international', 'education,unicef', 'skill', 'unicef,education',
                                            'executive', 'women', 'ministry', 'go', 'education,skill',
                                            'ministry,education', 'challenge,education', 'lack', 'child',
                                            'bank,education', 'executive,education'],
            'Literacy and Education': ['literacy,education']
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def security(self, index, embeddings):

        clusters = {

            'Security': [
                'security',
                'security,security',
                'council,security',
                'forces,security',
                'protests,security',
                'services,security',
                'security,protests',
                'legislation,security',
                'territory,security',
                'hacking,security',
                'unsc,security',
                'breach,security',
                'security,threat'
            ],
            'Cybersecurity': [
                'cyber',
                'malware',
                'ransomware,hacker',
                'malware,software',
                'malware,phishing',
                'malware,malware',
                'ransomware'
            ],
            'Data and Breach': [
                'data',
                'breach',
                'breach,data',
                'theft',
                'theft,identity'
            ],
            'Risk and Industry': [
                'industry',
                'industry,risk',
                'sector',
                'sector,risk',
                'risk,banking',
                'risk,corporate',
                'risk,power',
                'risk,medical',
                'risk,critical',
                'risk,mining',
                'risks,software',
                'risks,bitcoin',
                'risks,programs',
                'devices,risk'
            ],
            'Council and Governance': [
                'council',
                'governance',
                'governance,risk',
                'governance,protection',
                'council,unsc'
            ],
            'Power and Infrastructure': [
                'power',
                'infrastructure',
                'protection,power',
                'plant,power'
            ],
            'Identity and Privacy': [
                'identity',
                'theft,identity'
            ],
            'Healthcare and Medical': [
                'medical',
                'risk,medical',
                'healthcare'
            ],
            'Physical and Property': [
                'physical',
                'property'
            ],
            'Threat and Critical': [
                'threats',
                'threat,critical',
                'protests,threat',
                'plant,threat',
                'territory,threat'
            ],
            'Plant and Protection': [
                'plant',
                'plant,risk',
                'plant,protection',
                'plant,power'
            ],
            'Digital and Technology': [
                'digital',
                'devices'
            ],
            'Aviation': [
                'aviation'
            ]
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def political(self, index, embeddings):

        clusters = {
            'Political Leaders': ['president', 'prime minister', 'minister'],
            'Political Parties': ['party', 'congress', 'bjp', 'national'],
            'Elections': ['presidential,election', 'election', 'elections', 'election campaign', 'election party',
                          'state election', 'election president', 'election state', 'election congress', 'election bjp',
                          'election polls', 'assembly election', 'polls election', 'election prime',
                          'bjp,congress,election'],
            'Government Bodies': ['assembly', 'commission', 'lok sabha', 'parliament', 'constitution'],
            'Political Policies': ['policy', 'foreign']
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data

    def economy(self, index, embeddings):
        clusters = {

            'Macroeconomic Indicators': [
                'economy',
                'inflation',
                'recession',
                'gdp',
                'global economy',
                'economic survey',
                'economic policy',
                'economic crisis',
                'fiscal policy',
                'hyperinflation',
                'economic challenges',
                'stock market crash',
                'agriculture sector',
                'industrial sector'
            ],
            'Government and Public Finance': [
                'government',
                'finance minister',
                'budget',
                'debt',
                'tax',
                'union budget',
                'finance ministry',
                'revenue',
                'act',
                'loan',
                'borrowing',
                'regulations',
                'fiscal policy',
                'finance commission',
                'government finances'
            ],
            'Monetary Policy and International Finance': [
                'monetary policy',
                'world bank',
                'international monetary fund',
                'sanctions',
                'export',
                'import',
                'supply chain',
                'banking sector',
                'agreements',
                'sustainable development',
                'foreign direct investment',
                'trade war'
            ],
            'Financial Institutions and Policies': [
                'capital',
                'committee',
                'rbi',
                'niti aayog'

            ],
            'Social Economic Issues': [
                'fuel prices',
                'price hikes',
                'violations',
                'money laundering',
                'food security'
            ]
        }
        if isinstance(self.data.at[index, 'Keywords'], str):
            keywords = [word.strip() for word in
                        self.data.at[index, 'Keywords'].split(',') + self.data.at[index, 'Keywords'].split()]
            assigned_clusters = []
            highest_similarity = 0.0
            most_similar_cluster = None

            for cluster, words in clusters.items():
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
            self.data.at[index, 'cluster'] = assigned_clusters_str

        return self.data
# df2..............
output=sub_buckets(df2)
updated_data = output.data
Final_Data_Frame=pd.DataFrame(updated_data)