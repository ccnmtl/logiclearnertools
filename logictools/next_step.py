import time
from logictools.expression_parser import validate_and_get_frontier


def next_step(next_expr, next_rule, step_list, target):
    """
    Takes in the user's next step, validates it, and returns a hint if needed.
    nextStep: str; user's proposed next step,
    nextRule: str or Enum; user's proposed rule,
    stepList: list; list of user's valid steps so far,
    target: str; target expression
    :return: {
        isValid: bool; whether expression is valid or not
        isSolution: bool; whether the target has been reached
        errorCode: Enum; if error
        errorMsg: str; if error
        nextFrontier: list; possible next steps
        hintExpression: next step hint
        hintRule: next rule hint
    }
    """
    cur_expr = step_list[-1]
    response = validate_and_get_frontier(cur_expr, next_expr, next_rule, target)

    # super hacky placeholder for search
    idx = int(time.time() % len(response["nextFrontier"]))

    hint = response["nextFrontier"][idx]
    response["hintExpression"], response["hintRule"] = hint

    return response
