import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold, LeaveOneOut, learning_curve, LearningCurveDisplay, RandomizedSearchCV, RepeatedKFold, ShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, PowerTransformer, StandardScaler


DPI = None
NUM_JOBS = 4
RANDOM_SEED = 42
TEST_SET_FRACTION = 0.2
# black, blue, deep blue, red, deep red from https://www.materialpalette.com
PLOT_COLOURS = ['#212121', '#3F51B5', '#303F9F', '#FF5252', '#D32F2F']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#303F9F', '#FF5252', '#D32F2F'])

FIG_DIR = '/home/DART/jonathant/Documents/iterative-archetypal-analysis/metal-nanoparticle-energy-regression/figs'
# FIG_DIR = '/scratch/q27/jt5911/metal-nanoparticle-causal-path-probability/figs'


def rmNullLowVarFeats(featsDF, rmNull=False, varThresh=0.0, verbose=False):
    if rmNull:
        if verbose:
            print(f"Removing the features with null values...")
            print(f"  Original number of features: {len(featsDF.columns)}")
        featsToDrop = []
        for feat in featsDF.columns:
            if featsDF[feat].isnull().any():
                featsToDrop.append(feat)
        featsDF.drop(featsToDrop, axis=1, inplace=True)
        if verbose:
            print(f"  Total number of features left: {len(featsDF.columns)}\n")
    # if verbose:
    #     print(f"Removing the features with null values...")
    #     print(f"  Original number of features: {len(featsDF.columns)}")
    # featsToDrop = []
    # for feat in featsDF.columns:
    #     if featsDF[feat].isnull().any():
    #         featsToDrop.append(feat)
    # featsDF.drop(featsToDrop, axis=1, inplace=True)
    # if verbose:
    #     print(f"  Total number of features left: {len(featsDF.columns)}\n")
    
    if verbose:
        print(f"Removing the features with variance below {varThresh:.2f}...")
        print(f"  Original number of features: {len(featsDF.columns)}")
    featsToDrop = []
    for feat in featsDF.columns:
        if featsDF[feat].var() <= varThresh:
            if verbose:
                print(f"    {feat}:    {featsDF[feat].var():.3f}")
            featsToDrop.append(feat)
    featsNoLowVarDF = featsDF.drop(featsToDrop, axis=1, inplace=False)
    if verbose:
        print(f"  Total number of features left: {len(featsNoLowVarDF.columns)}\n")
    return featsNoLowVarDF


def rmHighCorrFeats(featsDF, corrThresh=0.9, verbose=False):
    if verbose:
        print(f"Removing the second feature from every pair of features with correlation above {corrThresh:.2f}...")
        print(f"  Original number of features: {len(featsDF.columns)}")
    if corrThresh >= 1.0:
        if verbose:
            print(f"  Total number of features left: {len(featsDF.columns)}\n")
        return featsDF
    featsToDrop = []
    for (i, feat1) in enumerate(featsDF.columns):
        for (j, feat2) in enumerate(featsDF.columns):
            if feat1 == feat2: 
                continue
            if j <= i: 
                continue
            corr = featsDF[feat1].corr(featsDF[feat2])
            if abs(corr) >= corrThresh:
                if feat1 in featsToDrop or feat2 in featsToDrop:
                    continue
                print(f"    {feat1} {feat2}:    {abs(corr):.3f}")
                featsToDrop.append(feat2)
    featsNoHighCorrDF = featsDF.drop(featsToDrop, axis=1, inplace=False)
    if verbose:
        print(f"  Total number of features left: {len(featsNoHighCorrDF.columns)}\n")
    return featsNoHighCorrDF


def splitScaleData(df, polyFeats=False, scaling='minmax', isReg=True, verbose=False):
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    if polyFeats:
        X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)  # Polynomial features (optional)
        XNoLowVar = rmLowVarFeats(X, varThresh=0.0, verbose=verbose)
        X = rmHighCorrFeats(XNoLowVar, corrThresh=0.9, verbose=verbose)
    stratify = None if isReg else y
    Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=TEST_SET_FRACTION, random_state=RANDOM_SEED, 
                                                    shuffle=True, stratify=stratify)  # Train-test split

    if scaling:
        if verbose:
            print(f"Applying {scaling} scaling to feature values...")
        if scaling == 'minmax':
            scaler = MinMaxScaler() 
        elif scaling == 'standard':
            scaler = StandardScaler()
        elif scaling == 'yeo-johnson':
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            raise Exception(f"Unknown {scaling} scaling specified!")
        Xtrain = scaler.fit_transform(Xtrain)
        Xtrain = pd.DataFrame(Xtrain, columns=X.columns)
        Xtest = scaler.transform(Xtest)
        Xtest = pd.DataFrame(Xtest, columns=X.columns)
    return Xtrain, Xtest, yTrain, yTest


def evalModel(model, Xtrain, yTrain, Xtest, yTest, cv, isReg=True, hideWarnings=False):
    if hideWarnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xtrain, yTrain)
    else:
        model.fit(Xtrain, yTrain)
    yPredTrain = model.predict(Xtrain)
    yPredTest = model.predict(Xtest)
    if isReg:
        metric, metricStr = 'r2', 'Coefficient of determination'
    else:
        metric, metricStr = 'f1_weighted', 'Weighted F1 score'
    print(f"  Trained and tested using {metricStr}.")
    print(f"    Training score: {model.score(Xtrain, yTrain):.3f}")
    crossValScores = cross_val_score(model, Xtrain, yTrain, cv=cv, scoring=metric)
    print(f"    Cross-validation score: {crossValScores.mean():.3f} +/- {crossValScores.std():.3f}")
    print(f"    Testing score:")  # {regModel.score(Xtest, yTest):.3f}
    if isReg:
        print(f"      Mean absolute error: {mean_absolute_error(yTest, yPredTest):.6f}")
        print(f"      Mean squared error: {mean_squared_error(yTest, yPredTest):.6f}")
        print(f"      Root mean squared error: {root_mean_squared_error(yTest, yPredTest):.6f}")
        print(f"      Coefficient of determination: {r2_score(yTest, yPredTest):.3f}")
    else:
        print(classification_report(yTest, yPredTest, target_names=[f"Class {i}" for i in sorted(yTest.unique())], 
                                    digits=3, zero_division=0.0))
        print(f"      F1 score: {model.score(Xtest, yTest):.3f}")
    return yPredTrain, yPredTest, model


def plotR2(yTrain, yTest, yPredTrain, yPredTest, modelName, label,
           dataColour=PLOT_COLOURS[1], bestFitColour=PLOT_COLOURS[3], 
           bestFitLW=1, perfectFitLW=1.5, markerSize=20, markerAlpha=1.0, markerLW=0.7, 
           figSize=(7, 3), figName=f"{FIG_DIR}/R2.png", 
           showFig=False):
    fig, axes = plt.subplots(1, 2, figsize=figSize, dpi=DPI)

    perfectFitTrain = np.linspace(yTrain.min(), yTrain.max(), num=100)
    mTrain, cTrain = np.polyfit(yTrain, yPredTrain, deg=1)
    axes[0].scatter(yTrain, yPredTrain, 
                    color=dataColour, s=markerSize, alpha=markerAlpha, lw=markerLW, 
                    edgecolors=PLOT_COLOURS[0], zorder=2)
    axes[0].plot(perfectFitTrain, cTrain + mTrain*perfectFitTrain, 
                 color=bestFitColour, linestyle='--', lw=bestFitLW, zorder=3)
    axes[0].plot(perfectFitTrain, perfectFitTrain, 
                 color=PLOT_COLOURS[0], lw=perfectFitLW, zorder=1)
    axes[0].grid(linestyle='dotted')
    axes[0].legend(['Training Data', 'Best Fit', 'Perfect Fit'])

    perfectFitTest = np.linspace(yTest.min(), yTest.max(), num=100)
    mTest, cTest = np.polyfit(yTest, yPredTest, deg=1)
    axes[1].scatter(yTest, yPredTest, 
                    color=dataColour, s=markerSize, alpha=markerAlpha, lw=markerLW, 
                    edgecolors=PLOT_COLOURS[0], zorder=2)
    axes[1].plot(perfectFitTest, cTest + mTest*perfectFitTest, 
                 color=bestFitColour, linestyle='--', lw=bestFitLW, zorder=3)
    axes[1].plot(perfectFitTest, perfectFitTest, 
                 color=PLOT_COLOURS[0], lw=perfectFitLW, zorder=1)
    axes[1].grid(linestyle='dotted')
    axes[1].legend(['Testing Data', 'Best Fit', 'Perfect Fit'])

    fig.text(0.53, 0.0, r'Actual ' + label, ha='center')
    fig.text(0.0, 0.4, r'Predicted ' + label, ha='center', rotation=90)
    plt.tight_layout()
    plt.savefig(figName, bbox_inches='tight')
    if showFig:
        plt.show()


def plotR2All(yTrain, yTest, yPredTrain, yPredTest, modelName, label,
              trainDataColour=PLOT_COLOURS[1], trainBestFitColour=PLOT_COLOURS[2], 
              testDataColour=PLOT_COLOURS[3], testBestFitColour=PLOT_COLOURS[4], 
              bestFitLW=1.8, perfectFitLW=1.5, 
              markerSize=25, markerAlpha=1.0, markerLW=0.9, 
              figSize=(4.5, 3.5), figName=f"{FIG_DIR}/R2All.png", 
              showFig=False):
    fig, ax = plt.subplots(figsize=figSize, dpi=DPI)
    
    mTrain, cTrain = np.polyfit(yTrain, yPredTrain, deg=1)
    perfectFitTrain = np.linspace(yTrain.min(), yTrain.max(), num=100)
    plt.scatter(yTrain, yPredTrain, 
                color=trainDataColour, s=markerSize, alpha=markerAlpha, lw=markerLW, 
                edgecolors=PLOT_COLOURS[0], zorder=2)
    plt.plot(perfectFitTrain, cTrain + mTrain*perfectFitTrain, 
             color=trainBestFitColour, linestyle='--', lw=bestFitLW, zorder=3)

    mTest, cTest = np.polyfit(yTest, yPredTest, deg=1)
    perfectFitTest = np.linspace(yTest.min(), yTest.max(), num=100)
    plt.scatter(yTest, yPredTest, 
                color=testDataColour, s=markerSize, alpha=markerAlpha, lw=markerLW, 
                edgecolors=PLOT_COLOURS[0], zorder=2)
    plt.plot(perfectFitTest, cTest + mTest*perfectFitTest, 
             color=testBestFitColour, linestyle='--', lw=bestFitLW, zorder=3)

    perfectFit = np.linspace(min(yTrain.min(), yTest.min()), max(yTrain.max(), yTest.max()), num=100)
    plt.plot(perfectFit, perfectFit, color=PLOT_COLOURS[0], lw=perfectFitLW, zorder=1)
    
    plt.xlabel(r'Actual ' + label)
    plt.ylabel(r'Predicted ' + label)
    plt.grid(linestyle='dotted')
    plt.legend(['Training Data', 'Training Best Fit', 'Testing Data', 'Testing Best Fit', 'Perfect Fit'])
    plt.savefig(figName, bbox_inches='tight')
    if showFig:
        plt.show()


def hyperparamTune(estimator, paramGrid, Xtrain, yTrain, 
                   scoringMetric='r2', cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED), numIter=100,
                   hideWarnings=False):
    # search = GridSearchCV(estimator=estimator, param_grid=paramGrid, cv=cv, scoring=scoringMetric, 
    #                       n_jobs=1, error_score=np.nan, verbose=0)
    search = RandomizedSearchCV(estimator=estimator, n_jobs=-1, cv=cv, n_iter=numIter,
                                param_distributions=paramGrid, scoring=scoringMetric, 
                                random_state=RANDOM_SEED, error_score=np.nan, verbose=0)
    if hideWarnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searchResults = search.fit(Xtrain, yTrain)
    else:
        searchResults = search.fit(Xtrain, yTrain)
    bestModel = searchResults.best_estimator_
    print(f"Best score: {searchResults.best_score_:.3f}")
    print(f"Best parameter combination: {searchResults.best_params_}")
    return searchResults


def plotLearningCurve(estimator, Xtrain, yTrain, modelName, cv,
                      figSize=(4.5, 3.5), metric='Coefficient of determination', figName=f"{FIG_DIR}/LearnCurve.png", 
                      showFig=False, hideWarnings=False):
    fig, ax = plt.subplots(figsize=figSize, dpi=DPI)
    trainColour = "#303F9F"
    cvColour = "#D32F2F"
    scoring = 'r2' if metric == 'Coefficient of determination' else 'f1_weighted'  # neg_root_mean_squared_error, roc_auc_score_over_weighted, f1_weighted
    learningCurveParams = {'X': Xtrain, 
                           'y': yTrain, 
                           'train_sizes': np.linspace(0.1, 1.0, 10), 
                           'cv': cv, # ShuffleSplit(n_splits=50, test_size=TEST_SET_FRACTION, random_state=RANDOM_SEED), 
                           'score_type': 'both', 
                           'n_jobs': NUM_JOBS, 
                           'std_display_style': 'fill_between', 
                           'scoring': scoring,
                           'score_name': metric, 
                           'line_kw': {'marker': 'o'},
                           'random_state': RANDOM_SEED}
    if hideWarnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LearningCurveDisplay.from_estimator(estimator, **learningCurveParams, ax=ax)
    else:
        LearningCurveDisplay.from_estimator(estimator, **learningCurveParams, ax=ax)

    lines = ax.get_lines()
    if len(lines) >= 1:
        lines[0].set_color(trainColour)
        lines[0].set_markerfacecolor(trainColour)
        lines[0].set_markeredgecolor(trainColour)
    if len(lines) >= 2:
        lines[1].set_color(cvColour)
        lines[1].set_markerfacecolor(cvColour)
        lines[1].set_markeredgecolor(cvColour)

    fills = ax.collections
    if len(fills) >= 1:
        fills[0].set_facecolor(trainColour)
        fills[0].set_edgecolor("none")
    if len(fills) >= 2:
        fills[1].set_facecolor(cvColour)
        fills[1].set_edgecolor("none")

    handles, label = ax.get_legend_handles_labels()
    plt.xlabel('Number of training samples')
    plt.legend(handles[:2], ['Training', 'Cross-validation'])
    plt.grid(linestyle='dotted')
    plt.savefig(figName, bbox_inches='tight')
    if showFig:
        plt.show()


def plotTopImpFeats(model, Xtrain, modelName, 
                    isForest=False, numFeat=40, figSize=(13, 3), figName=f"{FIG_DIR}/FeatImp.png", 
                    showFig=False):
    featureNames = model.feature_names_in_ if modelName != 'LGB' else model.feature_name_
    if isForest:
        impStds = np.std([tree.feature_importances_ for tree in model.estimators_])
        featImps = pd.DataFrame({'feature': featureNames, 'importance': model.feature_importances_, 'std': impStds})
    elif 'ETR' in modelName:
        featImps = pd.DataFrame({'feature': featureNames, 'importance': model.estimator.feature_importances_})
    else:
        featImps = pd.DataFrame({'feature': featureNames, 'importance': model.feature_importances_})
    featImps.sort_values(by='importance', ascending=False, inplace=True)
    topFeats = featImps.iloc[:numFeat, :]

    fig, ax = plt.subplots(figsize=figSize, dpi=DPI)
    if isForest:
        plt.bar(x=topFeats['feature'], height=topFeats['importance'], yerr=impStds, width=0.9, color=PLOT_COLOURS[-1], edgecolor=PLOT_COLOURS[0], lw=1)
    else:
        plt.bar(x=topFeats['feature'], height=topFeats['importance'], width=0.9, color=PLOT_COLOURS[-1], edgecolor=PLOT_COLOURS[0], lw=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Feature importance')
    plt.grid(linestyle='dotted')
    plt.savefig(figName, bbox_inches='tight')
    if showFig:
        plt.show()


def runModel(Xtrain, yTrain, Xtest, yTest, cv, model, modelName, datasetName, figNameAppendix, numFeat=15,
             isReg=True, label='E_{tot}', showFeatCoef=True, plotFeatImp=False, verbose=False, showFig=False, hideWarnings=False):
    if verbose: 
        print(f"  Running {modelName}")
    yPredTrain, yPredTest, model = evalModel(model, Xtrain, yTrain, Xtest, yTest, cv, isReg=isReg, hideWarnings=hideWarnings)
    if isReg:
        metric, labelStr = 'Coefficient of determination', 'E'
        plotR2(yTrain, yTest, yPredTrain, yPredTest, modelName, label,
               figName=f"{FIG_DIR}/{datasetName}R2{modelName}_{figNameAppendix}.png", 
               showFig=showFig)
        plotR2All(yTrain, yTest, yPredTrain, yPredTest, modelName, label,
                  figName=f"{FIG_DIR}/{datasetName}R2All{modelName}_{figNameAppendix}.png", 
                  showFig=showFig)
    else:
        metric, labelStr = 'Weighted F1 score', 'surfPatt'
        fig, ax = plt.subplots(figsize=(4.2, 4))
        cm = ConfusionMatrixDisplay.from_estimator(model, Xtest, yTest, ax=ax)
        cm.figure_.savefig(f"{FIG_DIR}/{datasetName}{labelStr}ConfMat{modelName}_{figNameAppendix}.png", dpi=DPI, bbox_inches = 'tight')
    plotLearningCurve(model, Xtrain, yTrain, modelName, metric=metric, cv=cv,
                      figName=f"{FIG_DIR}/{datasetName}{labelStr}LearnCurve{modelName}_{figNameAppendix}.png", 
                      showFig=showFig, hideWarnings=hideWarnings)
    if showFeatCoef:
        print("  Feature coefficients:")
        for (i, coefficient) in enumerate(model.coef_):
            if round(coefficient, 3) >= 0.001:
                print(f"    {model.feature_names_in_[i]}: {coefficient:.3f}")
    if plotFeatImp:
        plotTopImpFeats(model, Xtrain, modelName, figSize=(11, 3),
                        figName=f"{FIG_DIR}/{datasetName}{labelStr}FeatImp{modelName}_{figNameAppendix}.png", 
                        showFig=showFig)
        plotTopImpFeats(model, Xtrain, modelName, numFeat=numFeat, figSize=(6, 3),
                        figName=f"{FIG_DIR}/{datasetName}{labelStr}FeatImpTop{numFeat}{modelName}_{figNameAppendix}.png", 
                        showFig=showFig)
    return yPredTrain, yPredTest, model
