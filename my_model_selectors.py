import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import copy


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on BIC scores
        # number of states :
        # n = np.sum(self.lengths)
        n = len((self.lengths))
        logN = np.log(n)

        min_bic = 1000000
        best_hmm_model = None
        for nb_hidden_state in range(self.min_n_components, self.max_n_components + 1):
            # model, logL = self.base_model(nb_hidden_state)
            try:
                hmm_model = GaussianHMM(n_components=nb_hidden_state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)

                # number of free states
                p = n * n + 2 * nb_hidden_state * n - 1
                bic = -2 * logL + p * logN
                if bic < min_bic:
                    min_bic = bic
                    best_hmm_model = hmm_model
            except:
                continue
        return best_hmm_model

        # raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_DIC_Score = float('-inf')
        M = len((self.words).keys())  # num of words

        best_hmm_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                LogLi = hmm_model.score(self.X, self.lengths)

            except:
                LogLi = float("-inf")

            SumLogL = 0

            for each_word in self.hwords.keys():
                X_each_word, lengths_each_word = self.hwords[each_word]

            try:
                SumLogL += hmm_model.score(X_each_word, lengths_each_word)

            except:
                SumLogL += 0

            DIC_Score = LogLi - (1 / (M - 1)) * (SumLogL - (0 if LogLi == float("-inf") else LogLi))

            if DIC_Score > best_DIC_Score:
                best_DIC_Score = DIC_Score
                best_hmm_model = hmm_model
        return best_hmm_model

        # raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_CV_Score = float('-inf')

        best_hmm_model = None
        if len(self.sequences) < 2:
            return None

        split_method = KFold(n_splits=2)

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            SumLogL = 0
            Counter = 0

            # for cv_train, cv_test in split_method.split(self.sequences):
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_train_idx))


                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                try:

                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)

                    LogL = hmm_model.score(X_test, lengths_test)

                    Counter += 1

                except:
                    LogL = 0

                SumLogL += LogL

            # AVG score
            CV_Score = SumLogL / (1 if Counter == 0 else Counter)
            # print("CV for %d components is %d" % (num_states, CV_Score))
            # Choose best model
            if CV_Score > best_CV_Score:
                best_CV_Score = CV_Score
                best_hmm_model = hmm_model

        return best_hmm_model
        # raise NotImplementedError
