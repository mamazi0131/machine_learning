from sklearn.model_selection import GridSearchCV

def parm_optimize(model, param_grid, scoring, cv, dataset ,is_all=False):
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
    grid_result = grid_search.fit(dataset.data, dataset.lable)
    print('Best: {:.3f} using {}'.format(grid_result.best_score_, grid_search.best_params_))
    if is_all:
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']
        for mean, param in zip(means, params):
            print('{:.3f}  with:  {}'.format(mean, param))