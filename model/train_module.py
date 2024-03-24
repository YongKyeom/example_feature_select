import os
import random

import numpy as np
import pandas as pd
import datetime

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from hyperopt import hp
from model.optimize_hyperparameter import fnOpt_HyperPara

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils.visualize import fnPrecision_Recall_Curve_Plot

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")


class TrainClassifier:
    '''
    RF, XGBoost, LightGBM 모델을 학습한 후 예측할 수 있는 모듈
    
    Args:
        train_data: 학습 Data
        test_data: 테스트 Data(Valid or Test)
        feature_ls: X변수 List
        target_nm: Y변수 명
        scale_flag: X변수들에 대한 Normalizing(Min-Max를 0~1사이로 변환)을 수행할 지 여부
        smote_flag: 데이터 불균형이 있을 경우 이를 해소하기 위해 학습Data를 SMOTE(Upsampling의 일종) 수행을 할 지 여부
        core_cnt: 병렬처리 시 사용할 CPU 갯수
        seed: Random seed
        verbose: Print process or not
    '''
    
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_ls: list,
        target_nm: str,
        scale_flag: bool,
        smote_flag: bool,
        core_cnt: int = -1,
        seed: int = 1000,
        verbose: bool = True
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.feature_ls = feature_ls
        self.target_nm = target_nm
        self.scale_flag = scale_flag
        self.smote_flag = smote_flag
        self.core_cnt = core_cnt
        self.seed = seed
        self.verbose = verbose
        
        if self.scale_flag is True:
            ## 변수 Scale
            scaler = MinMaxScaler()
            train_scale = scaler.fit_transform(self.train_data[self.feature_ls])
            test_scale = scaler.transform(self.test_data[self.feature_ls])

            ## DataFrame 변환
            train_scale = pd.DataFrame(
                train_scale,
                columns = self.feature_ls
            )
            train_scale[self.target_nm] = self.train_data[self.target_nm].values
            test_scale = pd.DataFrame(
                test_scale,
                columns = featureList
            )
            if self.target_nm in self.test_data.columns:
                ## 검증 Data
                test_scale[self.target_nm] = self.test_data[self.target_nm].values
            else:
                ## Test Data
                test_scale[self.target_nm] = None

            print('Scaled Train Info')
            print(train_scale.describe(), '\n')
            print('Scaled Test Info')
            print(test_scale.describe(), '\n')
            
            ## Redefine Train/Test Data
            self.train_data = train_scale
            self.test_data = test_scale
            
            del train_scale
            del test_scale
        
        if smote_flag is True:
            ## SMOTE 선언
            smote = SMOTE(random_state = self.seed)
            ## Resampling
            train_resample_x, train_resample_y = smote.fit_resample(
                self.train_data[self.feature_ls],
                self.train_data[self.target_nm]
            )
            ## Redefine TrainData
            self.train_data = train_resample_x
            self.train_data[TARGET_NM] = train_resample_y.values

            print('After OverSampling, the shape of TrainData: {} \n'.format(train_resample_x.shape))
            print("After OverSampling, counts of label '1': {}".format(sum(train_resample_y == 1)))
            print("After OverSampling, counts of label '0': {}".format(sum(train_resample_y == 0)), '\n')
            
            del train_resample_x
            del train_resample_y
        
    def fnRedefinFeatureList(self, new_feature_ls):
        '''
        Class 외부에서 변수선택법 적용 등으로 변수를 변경할 경우 사용
        '''
        self.feature_ls = new_feature_ls
    
    def fnOptimizingHyperPara(self):
        '''
        Bayesian Optimizing for Hyper-Parameter with TPE
        '''
        
        if len(self.para_space) == 0:
            ## 빈 Dictionary가 입력될 경우 Default Parameter 세팅
            self.best_para = {}
        else:
            ## Bayesian Optimizing with TPE
            (
                self.trial_result,
                self.best_para
            ) = fnOpt_HyperPara(
                    total_data = self.train_data, 
                    x_var = self.feature_ls, 
                    y_var = self.target_nm, 
                    space = self.para_space[self.model_nm], 
                    lean_rate_ls = self.learing_rate_ls, 
                    ml_model = self.model_nm, 
                    core_cnt = self.core_cnt, 
                    cv_num = self.cv_num, 
                    max_evals = self.max_evals, 
                    seed = self.seed, 
                    verbose = self.verbose,
                )
    
    def fnFit(
        self,
        model_nm: str,
        para_tune_flag: bool,
        para_space: dict,
        learing_rate_ls: list = [0.001, 0.01, 0.1],
        cut_off: float = 0.5,
        max_evals: int = 50,
        cv_num: int = 3,
    ):
        '''
        모델의 종류를 입력값으로 받아 Hyper-parameter 최적화 및 예측모델을 학습하는 메소드
        
        Args:
            model_nm: 예측모델(['rf', 'xgb', 'lgbm'] 중 하나)`
            para_tune_flag: 하이퍼 파라미터 최적화를 수행할 지 여부
            para_space: 최적화할 파라미터들의 공간, 빈 딕셔너리로 입력할 경우 최적화 없이 Default 파라미터로 적용
            learing_rate_ls: XGBoost, LGBM의 경우 파라미터 최적화를 할 경우 리스트 형태로 입력
            cut_off: if Prob > cut_off then 1, else 0. None을 입력할 경우 Valid 데이터셋에서 최적값을 찾고, 숫자를 입력할 경우 해당 값을 사용
            max_eval: Hyper-parameter 최적화를 할 경우 최대 Try 횟수`
            cv_num: Number of Cross-validation
        '''
        
        self.model_nm = model_nm
        self.para_tune_flag = para_tune_flag
        self.para_space = para_space
        self.learing_rate_ls = learing_rate_ls
        self.cut_off = cut_off
        self.max_evals = max_evals
        self.cv_num = cv_num
        
        ## Step 1) Hyper Parameter 최적화 수행
        if para_tune_flag is True:
            self.fnOptimizingHyperPara()
        else:
            self.best_para = para_space
        
        ## Step2) 예측모델 학습
        if self.model_nm == 'rf':
            self.model = RandomForestClassifier(**self.best_para)
            self.model.fit(
                self.train_data[self.feature_ls], 
                self.train_data[self.target_nm]
            )
        elif self.model_nm == 'xgb':
            self.model = XGBClassifier(**self.best_para)
            self.model.fit(
                X = self.train_data[self.feature_ls], 
                y = self.train_data[self.target_nm],
                early_stopping_rounds = 50,
                eval_set = [(self.train_data[self.feature_ls], self.train_data[self.target_nm])],
                verbose = False
            )
        else:
            self.model = LGBMClassifier(**self.best_para)
            self.model.fit(
                X = self.train_data[self.feature_ls], 
                y = self.train_data[self.target_nm],
                eval_set = [(self.train_data[self.feature_ls], self.train_data[self.target_nm])],
                callbacks = [
                    early_stopping(
                        stopping_rounds = 50,
                        verbose = False
                        )
                    ]
            )
        
        ## Step3) Cutoff 지점 최적화
        if self.cut_off is None:
            ## Precission, Recall 사이에서의 최적지점
            self.cut_off = fnPrecision_Recall_Curve_Plot(
                y_test = self.test_data[self.target_nm],
                pred_proba = self.model.predict_proba(self.test_data[self.feature_ls])[:, 1],
                plot_flag = self.verbose
            )
            print('Cut-off Value(Optimizing): {}'.format(self.cut_off))
        else:
            print('Cut-off Value(Default): {}'.format(self.cut_off))
    
    @staticmethod
    def fnPredict(
        model: None,
        test_data: pd.DataFrame,
        feature_ls: list,
        target_nm: str,
        cut_off: float
    ):
        '''
        모델을 예측하는 메소드
        클래스 외부 데이터에도 적용할 수 있도록 정적메소드로 구현
        
        Args:
            model: 학습이 완료된 예측모델
            test_data: 예측할 DataFrame
            feature_ls: X변수 List
            target_nm: Y변수 이름
            cut_off: Cutoff 지점(IF Prob > cut_off then 1 else 0)
        '''
        
        ## Predict
        predict_prob = model.predict_proba(test_data[feature_ls])
        predict_value = [1 if x > cut_off else 0 for x in predict_prob[:, 1]]
        
        return predict_value
    
    @staticmethod
    def fnScoreResult(
        y_value: list,
        predict_value: list,
        verbose: bool
    ):
        '''
        모델을 예측결과에 대한 Score 계산하는 메소드
        클래스 외부 데이터에도 적용할 수 있도록 정적메소드로 구현
        
        Args:
            y_value: 실제 Y값
            predict_value: 예측값
            verbose: Confusion Matrix를 Plot으로 출력할 지 여부
        '''
        ## Score
        score_acc = accuracy_score(y_value, predict_value)
        score_precision = precision_score(y_value, predict_value)
        score_recall = recall_score(y_value, predict_value)
        score_f1 = f1_score(y_value, predict_value)
        
        score_df = pd.DataFrame(
            {
                'ACCURACY': score_acc,
                'PRECISSION': score_precision,
                'RECALL': score_recall,
                'F1_SCORE': score_f1
            },
            index = [0]
        )
        confusionmatrix = confusion_matrix(y_value, predict_value)

        print('Score List')
        print(score_df, '\n')
        print('Confusion Matrix')
        print(confusionmatrix, '\n')
        print(classification_report(y_value, predict_value))

        if verbose is True:
            sns.heatmap(confusionmatrix, annot=True, cmap='Blues')
            plt.show()
        
        return score_df, confusionmatrix