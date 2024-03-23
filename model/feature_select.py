import os
import random
import datetime

from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

def fnFixSeed(seed: int = 1000):
    random.seed(seed)                        ## random
    np.random.seed(seed)                     ## numpy
    os.environ["PYTHONHASHSEED"] = str(seed) ## os

def fnFeatSelect_RandVar(
    df_x, 
    df_y, 
    x_var, 
    core_cnt, 
    rand_num = 10, 
    threshold = 0.00,
    n_estimators = 500, 
    seed = 1000
):
    """
       To select feature with random variables
       1. Add random variables(random sampling)
       2. Fit random forest
       3. Select feature which has more feature importance than random variables

           Args:
               df: Total data
               x_var: feature list of ml_model
               y_var: target variable's name
               rand_num: cnt of adding variables
               threshold: threshold of feature importance

           Returns:
               feature list(selected)
    """
    
    try:
        ## 학습 DataFrame
        df_rand     = df_x.copy()
        ## Ranom변수이름 List
        random_cols = []
        ## Seed 고정
        fnFixSeed(seed)
        
        for i0 in range(1, rand_num + 1):
            ## name of random feature
            random_col = '__random_{}__'.format(i0)

            ## add random feature
            df_rand[random_col] = np.random.rand(df_rand.shape[0])

            ## add random feature name
            random_cols.append(random_col)

        ## Fit RF
        model_rf = RandomForestClassifier(n_estimators = n_estimators, n_jobs = core_cnt, random_state = seed)
        model_rf.fit(df_rand[x_var + random_cols], df_y)

        ## Feature importance
        feat_imp_df = pd.DataFrame({
            'feature_name': x_var + random_cols,
            'feature_importance': model_rf.feature_importances_
            }
        )

        ## Sort feature importance
        feat_imp_df = feat_imp_df.sort_values('feature_importance', ascending = False).reset_index(drop = True)

        ## Importance of random features
        imp_random    = feat_imp_df.loc[feat_imp_df.feature_name.isin(random_cols), 'feature_importance'].values
        imp_threshold = max(np.percentile(imp_random, 50), threshold)

        ## Filter with imp_threshold
        feat_imp_filter = feat_imp_df[feat_imp_df['feature_importance'] > imp_threshold]

        ## Selet feature
        feat_select = list(set(feat_imp_filter.feature_name) - set(random_cols))
        feat_select.sort()

    except Exception as e:
        print('Error in fnFeatSelect_RandVar')
        print(e)
        
        raise Exception('Check error')

    return feat_select

def fnInitialize(n ,m):
    '''
    유전 알고리즘의 초기해를 만드는 함수
    변수선택은 이진 인코딩(선택O, 선택X)을 함으로 이를 random choice로 선택
    
    Args:
        n: 유전자 갯수
        m: 특징(변수) 갯수
    
    Returns:
        초기해
    '''
    
    ## Random으로 변수별 사용유무 지정
    itit_sol = np.random.choice([0, 1], (n-1, m))
    ## 변수 전체를 사용하는 유전자 추가
    all_use_sol = [1 for _ in range(m)]
    
    itit_sol = np.vstack([itit_sol, all_use_sol])
    
    return itit_sol

def fnFitness(
    x_df: pd.DataFrame, 
    y: list, 
    model: None, 
    feat_sol: list, 
    cv_num = 3, 
    scoring = 'neg_log_loss'
):
    '''
    유전알고리즘의 적합도를 계산하는 함수
    
    Args:
        x_df: 학습 DataFrame
        y: Target Value
        model: 적합도를 계산할 모델
        feat_sol: 변수별 사용여부 -> 해집합
        cv_num: Cross validation 수행 횟수
        scoring: 적합도를 계산할 Metric
    
    Returns:
        적합도 점수
    '''
    score = cross_val_score(model, x_df.iloc[:, feat_sol], y, cv = cv_num, scoring = scoring)
    
    return score.mean()

def fnSelect(z, s, k):
    '''
    유전알고리즘의 선택 연산자(부모세대에서 k개를 선택)
    선택 연산자는 룰렛 휠 방법을 사용
      => 탐색 공간이 매우 넓고, 적합도가 0과 1사이 이므로 적합도별로 차이가 크게 나기 어렵기 떄문에
         룰렛 휠을 쓰면 다양한 해를 탐색할 수 있음
    
    Args:
        z: 현재 세대의 해집합
        s: 현재 세대의 해집합별 Score
        k: 선택할 해의 갯수(k)
    
    Returns:
        적합도를 계산할 변수 리스트
    '''
    
    selected_index = []
    _s = s.copy()
    for _ in range(k):
        ## 유전자조합별 Score(Fitness)를 기반으로 한 추출확률 리스트
        probs = _s / _s.sum()
        ## 유전자조합 선택
        z_idx = np.random.multinomial(1, probs).argmax()
        ## 선택한 유전자 리스트업
        selected_index.append(z_idx)
        ## 선택한 유전자는 재선택이 되지 않도록 확률값을 0으로 변경
        _s[z_idx] = 0
    
    return z[selected_index]

def fnCrossOver(x1, x2):
    '''
    유전알고리즘의 교차 연산자(한 점 교차 연산자)
    
    Step
        1) 1과 len(x1) 사이의 한 점을 임의로 선택
        2) x1과 x2를 1)에서 선택한 지점까지 슬라이싱한 후 병합
    
    Args:
        x1: 부모 세대의 해집합 중 하나
        x2: 부모 세대의 해집합 중 하나
    
    Returns:
        부모 세대의 교집합
    '''
    
    point_idx = np.random.choice(range(1, len(x1)))
    new_x = np.hstack([x1[:point_idx], x2[point_idx:]])
    
    return new_x

def fnBitFlip(z, p):
    '''
    유전알고리즘의 돌연변이 연산자(비트 플립 돌연변이)
    
    Args:
        z: 자식세대 유전자(해집합)
        p: 돌연변이 확률
    
    Returns:
        돌연변이 유전자
    '''
    ## 해집합의 각 요소별 변이 확률
    probs = np.random.random(len(z))
    ## 각 요소별로 p확률로 사용여부를 반전(0 or 1이므로 1-z를 하면 0->1, 1->0으로 변환됨) 
    z[probs < p] = 1 - z[probs < p]
    
    return z

def fnFeatSelect_GeneticAlgo(
    n: int,
    feat_num: int,
    select_sol_num: int,
    p: float,
    q: float,
    num_generation: int,
    x_df: pd.DataFrame,
    y: list,
    model_instance: None,
    cv_num: int = 3, 
    scoring: str = 'neg_log_loss',
    seed: int = 1000
):
    '''
    유전알고리즘(Genetic Algorithm)을 활용한 변수선택을 수행하는 함수
    
    Args:
        n: 세대 당 해의 갯수
        feat_num: 특징 갯수
        select_sol_num: 세대별로 선택할 해의 갯수(K) => 반드시 n보다 작아야 함
        p: 유전 개체가 돌연변이가 될 확률
        q: 돌연변이 확률
        num_generation: 세대 수
        x_df: 학습 데이터
        y: 타겟 데이터
        model_instance: 적합도를 계산할 모델
        cv_num: 적합도(Fitness) 계산시 사용하는 cross_val_score인자로, CV(Cross validation) 횟수
        scoring: 적합도를 계산할 Metric
    Returns:
        best_features: 선택된 변수 리스트
        best_score: 최적 변수의 Cross validation 점수
    '''
    
    st_time = datetime.datetime.now()
    print('Genetic Algorithm for Feature Selection is Start')
    
    ## Seed 고정
    fnFixSeed(seed)

    ## best_score 초기화
    best_score = -999999999999999
    ## Feature List
    feature_ls = []
    ## Score List
    score_ls = []
    ## 초기해
    sol_generation = fnInitialize(n, feat_num)
    
    ## 세대를 반복하면서 적합도 계산
    pbar = tqdm(
        range(num_generation),
        desc = 'Genetic Algorithm for Feature Selection',
        ascii = ' =',
        leave = True
    )
    for idx in pbar:
        ## 해 평가
        score = np.array(
            [fnFitness(x_df, y, model_instance, sol) for sol in sol_generation]
        )
        ## 현재 최적 해집합의 Score
        current_best_score = score.max()
        ## 현재 최적 해집합
        current_best_features = sol_generation[score.argmax()]
        ## Score 저장
        score_ls.append(current_best_score)
        ## Feature List 저장
        feature_ls.append(
            x_df.columns[current_best_features.astype(bool)].tolist()
        )
        
        ## 최고 해집합 업데이트
        if current_best_score > best_score:
            best_score = current_best_score
            best_features = current_best_features
            
            if idx > 0:
                print(
                    f'{idx + 1}/{num_generation}: Feature Selection!(Count: {sum(best_features)})'
                )
                
            ## Print Best Score
            pbar.desc = f'Genetic Algorithm Score: "{best_score}"'
        
        ## K개 해 선택
        new_generation = fnSelect(sol_generation, score, select_sol_num)
        
        ## 교배 및 돌연변이 연산
        for _ in range(n - select_sol_num):
            ## 부모세대 2개 선택
            parent_idx = np.random.choice(range(select_sol_num), 2, replace = False)
            ## 선택된 부모세대 끼리 교배
            child = fnCrossOver(
                new_generation[parent_idx[0]], new_generation[parent_idx[1]]
            )
            
            ## q확률로 자식세대의 돌연변이 발생
            if np.random.random() < q:
                ## 각 유전개체가 돌연변이될 확률 p
                child = fnBitFlip(child, p)
            
            new_generation = np.vstack([new_generation, child])
        
        sol_generation = new_generation.astype(bool)
    
    end_time = datetime.datetime.now()
    print('Genetic Algorithm for Feature Selection Elapsed Time: {} Minutes'.format(
        round((end_time - st_time).seconds / 60, 2)
        )
    )
    
    return x_df.columns[best_features.astype(bool)].tolist(), score_ls, feature_ls
        
            


