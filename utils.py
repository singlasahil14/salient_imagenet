import numpy as np
import pandas as pd
from collections import defaultdict

def print_with_stars(print_str, total_count=115, prefix="", suffix="", star='*'):
    str_len = len(print_str)
    left_len = (total_count - str_len)//2
    right_len = total_count - left_len - str_len
    final_str = "".join([star]*(left_len)) + print_str + "".join([star]*(right_len))
    final_str = prefix + final_str + suffix
    print(final_str)
    
def topk_predictive_features(class_index, robust_model, robust_features, k=5):
    W = (robust_model.model.fc.weight).detach().cpu().numpy()
    W_class = W[class_index: class_index + 1, :]
    FI_values = np.mean(robust_features * W_class, axis=0)

    features_indices = np.argsort(-FI_values)[:k]
    return features_indices

class MTurk_Results:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(self.csv_path)
        
        self.aggregate_results(self.dataframe)
        
        self.class_feature_maps()
        self.core_spurious_labels_dict()
        self.spurious_feature_lists()

        
    def aggregate_results(self, dataframe):
        answers_dict = defaultdict(list)
        reasons_dict = defaultdict(list)
        feature_rank_dict = defaultdict(int)
        wordnet_dict = defaultdict(str)
        
        for row in dataframe.iterrows():
            index, content = row
            WorkerId = content['WorkerId']

            class_index = int(content['Input.class_index'])
            feature_index = int(content['Input.feature_index'])
            feature_rank = int(content['Input.feature_rank'])
            
            wordnet_dict[class_index] = content['Input.wordnet_id']
            
            key = str(class_index) + '_' + str(feature_index)

            main_answer = content['Answer.main']
            confidence = content['Answer.confidence']
            reasons = content['Answer.reasons']
            
            answers_dict[key].append((WorkerId, main_answer, confidence, reasons))
            reasons_dict[key].append(reasons)
                
            feature_rank_dict[key] = feature_rank

        self.answers_dict = answers_dict
        self.feature_rank_dict = feature_rank_dict
        self.reasons_dict = reasons_dict
        self.wordnet_dict = wordnet_dict
    
    def core_spurious_labels_dict(self):
        answers_dict = self.answers_dict
        
        core_features_dict = defaultdict(list)
        spurious_features_dict = defaultdict(list)
        
        core_spurious_dict = {}
        core_list = []
        spurious_list = []
        for key, answers in answers_dict.items():
            class_index, feature_index = key.split('_')
            class_index, feature_index = int(class_index), int(feature_index)
            
            num_spurious = 0
            for answer in answers:
                main_answer = answer[1]
                if main_answer in ['separate_object', 'background']:
                    num_spurious = num_spurious + 1                

            if num_spurious >= 3:
                spurious_features_dict[class_index].append(feature_index)
                core_spurious_dict[key] = 'spurious'
                spurious_list.append(key)
                
            else:
                core_features_dict[class_index].append(feature_index)
                core_spurious_dict[key] = 'core'
                core_list.append(key)
                
        self.core_spurious_dict = core_spurious_dict
        self.core_list = core_list
        self.spurious_list = spurious_list
        
        self.core_features_dict = core_features_dict
        self.spurious_features_dict = spurious_features_dict
    
    def spurious_feature_lists(self):
        answers_dict = self.answers_dict
        
        background_list = []
        separate_list = []
        ambiguous_list = []
        for key, answers in answers_dict.items():
            num_background = 0
            num_separate = 0
            for answer in answers:
                main_answer = answer[1]
                if main_answer == 'background':
                    num_background = num_background + 1
                elif main_answer == 'separate_object':
                    num_separate = num_separate + 1
                                
            if num_background >= 3:
                background_list.append(key)
            elif num_separate >= 3:
                separate_list.append(key)
            elif (num_background + num_separate) >= 3:
                ambiguous_list.append(key)
                
        self.background_list = background_list
        self.separate_list = separate_list
        self.ambiguous_list = ambiguous_list
        
        
    def class_feature_maps(self):
        answers_dict = self.answers_dict
        
        keys_list = answers_dict.keys()
        
        feature_to_classes_dict = defaultdict(list)
        class_to_features_dict = defaultdict(list)
        
        for key in keys_list:
            class_index, feature_index = key.split('_')
            class_index = int(class_index)
            feature_index = int(feature_index)
            
            feature_to_classes_dict[feature_index].append(class_index)
            class_to_features_dict[class_index].append(feature_index)
            
        self.class_to_features_dict = class_to_features_dict
        self.feature_to_classes_dict = feature_to_classes_dict
        