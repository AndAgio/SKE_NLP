from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .cart import CartPredictor, LeafConstraints, LeafSequence
from skenlp.utils import get_default_random_seed
from ..schema import GreaterThan, DiscreteFeature
from ..logic import create_variable_list, create_head, create_term
from tuprolog.core import clause, Var, Struct
from tuprolog.theory import Theory, mutable_theory
from typing import Iterable, Union
import pandas as pd

TREE_SEED = get_default_random_seed()


class Cart:

    def __init__(self, max_depth: int = None, max_leaves: int = None,
                 discretization: Iterable[DiscreteFeature] = None,
                 normalization=None, simplify: bool = True):
        self.discretization = [] if discretization is None else list(discretization)
        self.normalization = normalization
        self._cart_predictor = CartPredictor(normalization=normalization)
        self.depth = max_depth
        self.leaves = max_leaves
        self._simplify = simplify

    def _create_body(self, variables: dict[str, Var], constraints: LeafConstraints) -> Iterable[Struct]:
        results = []
        for feature_name, constraint, value in constraints:
            features = [d for d in self.discretization if feature_name in d.admissible_values]
            feature: DiscreteFeature = features[0] if len(features) > 0 else None
            results.append(create_term(variables[feature_name], constraint) if feature is None else
                           create_term(variables[feature.name],
                                       feature.admissible_values[feature_name],
                                       isinstance(constraint, GreaterThan)))
        return results

    @staticmethod
    def _simplify_nodes(nodes: list) -> Iterable:
        simplified = [nodes.pop(0)]
        while len(nodes) > 0:
            first_node = nodes[0][0]
            for condition in first_node:
                if all([condition in [node[0] for node in nodes][i] for i in range(len(nodes))]):
                    [node[0].remove(condition) for node in nodes]
            simplified.append(nodes.pop(0))
        return simplified
    
    def set_valuable_tokens(self, valuable_tokens):
        self._cart_predictor.set_valuable_tokens(valuable_tokens)

    def get_valuable_tokens(self):
        return self._cart_predictor.get_valuable_tokens()

    def _create_theory(self, data: pd.DataFrame, mapping: dict[str: int], sort: bool = True) -> Theory:
        new_theory = mutable_theory()
        nodes = [node for node in self._cart_predictor]
        nodes = Cart._simplify_nodes(nodes) if self._simplify else nodes
        for (constraints, prediction) in nodes:
            if self.normalization is not None:
                m, s = self.normalization[data.columns[-1]]
                prediction = prediction * s + m
            if mapping is not None and prediction in mapping.values():
                for k, v in mapping.items():
                    if v == prediction:
                        prediction = k
                        break
            variables = create_variable_list(self.discretization, data, sort)
            new_theory.assertZ(
                clause(
                    create_head(data.columns[-1], list(variables.values()), prediction),
                    self._create_body(variables, constraints)
                )
            )
        return new_theory

    def extract(self, data: pd.DataFrame, mapping: dict[str: int] = None, sort: bool = True) -> Theory:
        self._cart_predictor.predictor = DecisionTreeClassifier(random_state=TREE_SEED) \
            if isinstance(data.iloc[0, -1], str) or mapping is not None \
            else DecisionTreeRegressor(random_state=TREE_SEED)
        if mapping is not None:
            data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: mapping[x] if x in mapping.keys() else x)
        self._cart_predictor.predictor.max_depth = self.depth
        self._cart_predictor.predictor.max_leaf_nodes = self.leaves
        self._cart_predictor.predictor.fit(data.iloc[:, :-1], data.iloc[:, -1])
        return self._create_theory(data, mapping, sort)

    def predict(self, data: pd.DataFrame, mapping: dict[str: int] = None) -> Iterable:
        ys = self._cart_predictor.predict(data)
        if mapping is not None:
            inverse_mapping = {v: k for k, v in mapping.items()}
            ys = [inverse_mapping[y] for y in ys]
        return ys

    def predict_and_explain(self, sample: pd.DataFrame):
        n_nodes = self._cart_predictor.predictor.tree_.node_count
        feature = self._cart_predictor.predictor.tree_.feature
        threshold = self._cart_predictor.predictor.tree_.threshold
        assert sample.shape[0] == 1
        node_indicator = self._cart_predictor.predictor.decision_path(sample)
        leaf_id = self._cart_predictor.predictor.apply(sample)
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[0 + 1]]
        activated_rules = 'Rule: '.format()
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[0] == node_id:
                continue
            # check if value of the split feature for sample 0 is below threshold
            if sample.iloc[0, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            # Append term to the activated rule
            activated_rules += "{feature}, ".format(feature='¬ ' + sample.columns[feature[node_id]]
                                                    if threshold_sign == '<='
                                                    else sample.columns[feature[node_id]])
        return activated_rules.strip(', '), self.predict(sample)

    @property
    def n_rules(self) -> int:
        return self._cart_predictor.n_leaves

    @property
    def predictor(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        return self._cart_predictor.predictor
