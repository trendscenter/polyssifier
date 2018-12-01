import unittest
from abc import ABC, abstractmethod
import sklearn.model_selection

class PolysisTests(object):

    def __init__(self, Analyzer, data, label):
        self.data = data
        self.label = label
        self.analyzer = Analyzer()

    def test_proper_subclass(self):
        self.assertTrue(issubclass(AnalyzerTest, Polysis))

    def test_initialize_folds(self):
        self.analyzer.initialize_folds()
        self.assertTrue(issubclass(self.analyzer.stratified_k_fold,
                        sklearn.model_selection._split._BaseKFold))

    def test_initialize_models(self):
        self.analyzer.initialize_models()
        pass

    def test_gather_results(self):
        self.analyzer.gather_results()
        pass

    def test_feature_importance(self):
        self.analyzer.feature_importance()
        pass

    def test_scorer(self):
        self.analyzer._scorer()
        pass

    def test_aggregate_results(self):
        self.analyzer.aggregate_results()
        pass

    def test_create_project(self):
        self.analyzer.create_project()
        pass
    
    def test_initialize_scores(self):
        self.analyzer.initialize_scores()
        pass

    def test_initialize_predictions(self):
        self.analyzer.initialize_predictions()
        pass

    def test_initialize_probabilities(self):
        self.analyzer.initialize_probabilites()
        pass

    def test_finalize_folds(self):
        self.analyzer.finalize_folds()
        pass

    def test_build(self):
        self.analyzer.build()
        pass

    def test_run(self):
        self.analyzer.run()
        pass

    def test_score_model(self):
        self.analyzer.score_model()
        pass

    def test_get_xy(self):
        self.analyzer.get_xy()
        pass

    def test_fit_model(self):
        self.analyzer.fit_model()
        pass

    def test_fit_model(self):
        self.analyzer.fit_model()
        pass

    def test_generate_report(self):
        self.analyzer.generate_report()
        pass

    def test_save_results(self):
        self.analyzer.save_results()
        pass

    def test_print_scores(self):
        self.analyzer.print_scores()
        pass

    def test_process_args(self):
        self.analyzer.process_args()
        pass

    def test_fit_all_models(self):
        self.analyzer.fit_all_models()
        pass

    def test_save_object(self):
        self.analyzer.save_object()
        pass

    def test_create_polynomial(self):
        self.analyzer.create_polynomial()
        pass

    def test_format_report_summary(self):
        self.analyzer.format_report_summary()
        pass