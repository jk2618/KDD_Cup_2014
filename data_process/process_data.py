import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ProcessNumericFeatures:
    '''
    input: features_numeric_raw (defined below)
    output: features_numeric_processed (DataFrame)
    '''
    def __init__(self, X):
        self.X = X
    
    def transform(self, X, features_numeric, scale=True):
        self.X = X.loc[:, features_numeric]
        if scale == True:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            import pandas as pd
            self.X = pd.DataFrame(self.X)
            return self.X
        else:
            self.X = pd.DataFrame(self.X)
            return self.X

class ProcessCategoryFeatures:
    '''
    input: features_category_raw (defined below)
    output: features_category_processed (DataFrame)
    '''
    def __init__(self, X):
        self.X = X
        
    def transform(self, X, features_category):
        self.X = X.loc[:, features_category]
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in self.X.columns.values:
            data = self.X.loc[:, col]
            le.fit(data.astype('str'))
            self.X.loc[:, col] = le.transform(self.X.loc[:, col].astype('str'))
        self.X2 = pd.get_dummies(self.X, columns = features_category, drop_first = True)
        return self.X2

def preprocessor(text): 
    import re
    from nltk.stem import SnowballStemmer
    processed = re.sub(r'[#|\!|\-|\+|:|//|\']', "", text)
    processed = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', ' ', processed).strip()
    processed = re.sub('[\s]+', ' ', processed).strip()
    processed = " ".join([SnowballStemmer("english").stem(word)
                          for word in processed.split()])
    return processed    
    
class ProcessTextFeatures:
    '''
    input: features_text_raw (defined below)
    output: features_text_processed (DataFrame)
    '''
    def __init__(self, X):
        self.X = X
          
    def transform(self, X, features_text = 'essay', dimen_reduc = 150):
        self.X = X.loc[:, features_text]
        
        import numpy as np
        separ = np.array_split(self.X.values, 20)
        result = np.array([])
        for i in range(20):
            result = np.concatenate((result, separ[i]), axis = 0)
        
        import re
        from nltk.stem import SnowballStemmer
        from sklearn.feature_extraction.text import TfidfVectorizer
        text_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                          ngram_range=(1, 2),
                                          preprocessor=preprocessor,
                                          stop_words='english')
        text_vectorizer.fit(result.astype('U'))
        self.X2 = text_vectorizer.transform(result.astype('U'))
        
        from sklearn.decomposition import TruncatedSVD
        tfidf_lsa = TruncatedSVD(n_components = dimen_reduc)
        self.X3 = tfidf_lsa.fit_transform(self.X2)
        import pandas as pd
        self.X3 = pd.DataFrame(self.X3)
        return self.X3

def get_processed_data(fraction = 0.2):
    os.chdir('C:/data/')
    essays = pd.read_csv("essays.csv")
    outcomes = pd.read_csv("outcomes.csv")
    projects = pd.read_csv("projects.csv")

    # merge the three datasets
    process1 = pd.merge(projects, outcomes, on='projectid', how='left', indicator=True)
    process1 = process1.loc[process1._merge=='both']
    process1 = process1.drop('_merge', axis=1)

    process2 = pd.merge(process1, essays, on='projectid', how='left', indicator=True)
    process2 = process2.loc[process2._merge=='both']
    process2 = process2.drop('_merge', axis=1)

    # take training sample between 2010-04-14 and 2014-04-01 (as requested by DonorsChoose)
    process3 = process2[(process2.date_posted > '2010-04-14') & (process2.date_posted < '2014-04-01')]
    process4 = process3.sample(frac=fraction, replace=False, random_state=1111)
    process4['essay_length'] = process4.essay.apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    process4['fully_funded'] = (process4.fully_funded=='t')
    process4['eligible_double_your_impact_match'] = (process4.eligible_double_your_impact_match=='t')
    process4['eligible_almost_home_match'] = (process4.eligible_almost_home_match=='t')

    # the missing values for our features are reletively small (0.5%), hence drop them for simplicity
    # otherwise one needs to impute missing values
    data_raw = process4.dropna(axis=0, how='any')

    features_numeric = ['total_price_including_optional_support','students_reached','great_messages_proportion','teacher_referred_count',
                        'fully_funded', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'essay_length']
    features_category = ['school_state','school_metro','teacher_teach_for_america','primary_focus_subject','resource_type',
                         'poverty_level','teacher_prefix']
    features_text = ['essay']

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(data_raw.is_exciting)

    # separate data into three parts
    features_numeric_raw = data_raw.loc[:, features_numeric]
    features_category_raw = data_raw.loc[:, features_category]
    features_text_raw = data_raw.loc[:, features_text]

    PNF = ProcessNumericFeatures(features_numeric_raw)
    features_numeric_processed = PNF.transform(features_numeric_raw, features_numeric)

    PCF = ProcessCategoryFeatures(features_category_raw)
    features_category_processed = PCF.transform(features_category_raw, features_category)

    PTC = ProcessTextFeatures(features_text_raw)
    features_text_processed = PTC.transform(features_text_raw)

    features_numeric_processed.reset_index(drop=True, inplace=True)
    features_category_processed.reset_index(drop=True, inplace=True)
    features_text_processed.reset_index(drop=True, inplace=True)

    # combine processed data
    features_processed_all = pd.concat([features_numeric_processed, features_category_processed], axis=1)
    
    return features_processed_all, labels