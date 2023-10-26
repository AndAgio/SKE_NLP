import re
from tuprolog.core import Clause
from tuprolog.theory import Theory
from tuprolog.core.operators import DEFAULT_OPERATORS, operator, operator_set, XFX
from tuprolog.core.formatters import TermFormatter
from statistics import mean


OP_IN = operator('in', XFX, 700)

OP_NOT = operator('not_in', XFX, 700)

RULES_OPERATORS = DEFAULT_OPERATORS + operator_set(OP_IN, OP_NOT)


def get_clauses(theory: Theory) -> list[Clause]:
    return [clause for clause in theory]


def theory_length(theory: Theory) -> int:
    return len(get_clauses(theory))


def get_clauses_lengths(theory: Theory) -> list:
    formatter = TermFormatter.prettyExpressions(True, RULES_OPERATORS)
    return [len(str(formatter.format(clause.body)).split(',')) for clause in theory]


def theory_cumbersomeness(theory: Theory) -> int:
    return mean(get_clauses_lengths(theory))


def get_concepts(theory: Theory) -> list:
    formatter = TermFormatter.prettyExpressions(True, RULES_OPERATORS)
    clauses = get_clauses(theory)
    concepts = []
    for clause in clauses:
        terms = str(formatter.format(clause.body)).split(',')
        for term in terms:
            splits = re.split('>= |=< |< |>', term)
            element = splits[0].lstrip().rstrip()
            if element not in concepts:
                concepts.append(element)
    return concepts


def theory_spread(theory: Theory) -> int:
    return len(get_concepts(theory))


def theory_complexity(theory: Theory) -> float:
    print('\n\nTHEORY COMPLEXITY:\n')
    print('Length = {}'.format(theory_length(theory)))
    print('Cumbersomeness = {}'.format(theory_cumbersomeness(theory)))
    print('Spread = {}'.format(theory_spread(theory)))
    return theory_length(theory) + theory_cumbersomeness(theory) + theory_spread(theory)
