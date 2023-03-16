import json
import os.path
from time import asctime, localtime, time

from logictools.AI.astar_heuristics import GeneHeuristic
from logictools.AI.astar_search import astar_search
from logictools.expression_parser import get_frontier
from logictools.logic_rule_transforms import search_operations


def get_questions():
    with open(os.path.join("..", "questions.json"), "r") as qf:
        questions = json.load(qf)['questions']
    return questions


def evaluate_against_question_bank(
        heuristic, results_file="heuristic_test_results.txt", max_timeout=5):
    with open(results_file, "w") as rf:
        est = asctime(localtime())
        rf.write(
            "A* Search Test: Heuristic: {}, Max Timeout: {} seconds\n\n"
            .format(est, heuristic.__name__, max_timeout)
        )

    def frontier_func(x):
        fr = get_frontier(
            x[0],
            include_paren=False,
            allowed_ops=search_operations)
        return fr

    def goal_func(x, target):
        return x[0] == target[0]

    questions = get_questions()
    num_solved = 0
    for i, q in enumerate(questions):
        start = q['premise']
        goal = q['target']
        start_time = time()
        is_solved, result = astar_search(
            start, goal, heuristic, frontier_func, goal_func, max_timeout=max_timeout)
        end_time = time()
        with open(results_file, "a") as rf:
            info_str = "{}. Premise: {}, Target: {}. ".format(
                i + 1, start, goal)
            if is_solved:
                solve_time = end_time - start_time
                rf.write(
                    info_str +
                    f"Solved in {solve_time:.4f} seconds. Solution: {result}\n.")
                print(
                    info_str +
                    f"Solved in {solve_time:.4f} seconds. Solution: {result}")
                num_solved += 1
            else:
                if result:
                    rf.write(
                        info_str +
                        f" Timeout occurred. Path to best node: {result}\n.")
                    print(
                        info_str +
                        f" Timeout occurred. Path to best node: {result}")
                else:
                    rf.write(info_str + " Error occurred.")
                    print(info_str + " Error occurred.")
    with open(results_file, "a") as rf:
        rf.write("\nSolved {}/{} questions.".format(num_solved, len(questions)))
        print("\nSolved {}/{} questions.".format(num_solved, len(questions)))


def frontier_func(x):
    fr = get_frontier(x[0], include_paren=False, allowed_ops=search_operations)
    return fr


def goal_func(x, target):
    return x[0] == target[0]


def get_heuristic_score(heuristic, questions, max_timeout=5):
    return sum([is_solved for is_solved, result in
                [astar_search(q["premise"], q["target"], heuristic, frontier_func, goal_func, max_timeout=max_timeout)
                 for q in questions]])


if __name__ == "__main__":
    gh = GeneHeuristic()
    gh.load("astar_heuristic_weights.txt")
    # evaluate_against_question_bank(gh.gene_meta_dist)

    questions = get_questions()
    score = get_heuristic_score(gh.gene_meta_dist, questions, max_timeout=5)
    print(f"Score: {score}")
