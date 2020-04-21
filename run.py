import argparse
import tempfile
import catboost
import numpy as np
from scipy.stats import uniform, loguniform
from sklearn.model_selection import PredefinedSplit


def tune_by_cv(train_path, cd_path, params_space, n_iter):
    clf = catboost.CatBoostClassifier()
    train_pool = catboost.Pool(train_path, column_description=cd_path)
    
    rs = clf.randomized_search(params_space, train_pool, cv=3, n_iter=n_iter, search_by_train_test_split=False, calc_cv_statistics=False, refit=False, verbose=False)
    
    return rs['params']


def tune_by_val(train_path, val_path, cd_path, params_space, n_iter):
    clf = catboost.CatBoostClassifier()
    train_pool = catboost.Pool(train_path, column_description=cd_path)
    val_pool = catboost.Pool(val_path, column_description=cd_path)
    
    with tempfile.NamedTemporaryFile(buffering=0) as out:
        with open(train_path, 'rb') as f:
            out.write(f.read())
        with open(val_path, 'rb') as f:
            out.write(f.read())
        merged_pool = catboost.Pool(out.name, column_description=cd_path)
    val_fold = [-1] * train_pool.shape[0] + [0] * val_pool.shape[0]
    splitter = PredefinedSplit(val_fold)
    assert len(val_fold) == merged_pool.shape[0]
    
    rs = clf.randomized_search(params_space, merged_pool, cv=splitter, n_iter=n_iter, search_by_train_test_split=False, calc_cv_statistics=False, refit=False, verbose=False)
    
    return rs['params']


def eval_on_test(train_path, val_path, test_path, cd_path, params):
    clf = catboost.CatBoostClassifier()
    clf.set_params(**params)
    train_pool = catboost.Pool(train_path, column_description=cd_path)
    val_pool = catboost.Pool(val_path, column_description=cd_path)
    test_pool = catboost.Pool(test_path, column_description=cd_path)
    
    clf.fit(train_pool, eval_set=val_pool, verbose=False)
    
    results = clf.eval_metrics(test_pool, metrics=['Logloss', 'Accuracy'])
    print("Test logloss:", results['Logloss'][-1])
    print("Test accuracy:", results['Accuracy'][-1])


def compare_tuning_cv_vs_val(train_path, val_path, test_path, cd_path):
    params_space = {
        'learning_rate': loguniform(np.exp(-5), 1),
        'random_strength': range(1, 21),
        'l2_leaf_reg': loguniform(1, 10),
        'subsample': uniform(0.2, 1 - 0.2),
        'leaf_estimation_iterations': range(1, 11)
    }
    n_iter = 20

    print("Searching for best parameters by CV...")
    best_params_cv = tune_by_cv(train_path, cd_path, params_space, n_iter)
    print("Searching for best parameters by validation set...")
    best_params_val = tune_by_val(train_path, val_path, cd_path, params_space, n_iter)

    print("Test results for parameters tuned by CV:")
    eval_on_test(train_path, val_path, test_path, cd_path, best_params_cv)
    print("Test results for parameters tuned by validation set:")
    eval_on_test(train_path, val_path, test_path, cd_path, best_params_val)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', dest='train_path', required=True)
    parser.add_argument('--val-path', dest='val_path', required=True)
    parser.add_argument('--test-path', dest='test_path', required=True)
    parser.add_argument('--cd-path', dest='cd_path', required=True)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    compare_tuning_cv_vs_val(args.train_path, args.val_path, args.test_path, args.cd_path)
