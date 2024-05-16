from sklearn.model_selection import GridSearchCV

def grid_search(model, X_train, y_train):
    
    param_grid = {'epochs': [10, 15, 20],
                'batch_size': [16, 32, 64]}

    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, n_jobs = -1,
                                scoring = 'accuracy', verbose = 1)

    grid_result = grid_search.fit(X_train, y_train)

    print(f'Best Accuracy for {grid_result.best_score_} using the parameters {grid_result.best_params_}')

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean = {mean:.4}, std = {stdev:.4} using {param}')

    best_estimator = grid_result.best_estimator_

    return best_estimator