import time

from logictools.AI.astar_heuristics import *
import json
from heapq import heappush, heappop
import logictools.logic_rule_transforms as lrt

inf = float('inf')


def astar_search(start, goal, neighbor_dist, frontier_func, goal_func,
                 goal_heuristic=None, max_depth=None, max_timeout=None, *args, **kwargs):
    """
    A* best first search algorithm
    :param start:
    :param goal:
    :param neighbor_dist:
    :param goal_heuristic:
    :param max_depth:
    :param max_timeout:
    :param args:
    :param kwargs:
    :return: tuple of <isfound, bestval>. Bestval is solution if isfound is True, else path to highest scoring node
    """

    if max_timeout:
        start_time = time.time()

    if not goal_heuristic:  # no separate heuristic for neighbor comparison or goal search
        goal_heuristic = neighbor_dist

    class SearchNode:
        def __init__(self, data, fscore=inf, gscore=inf):
            self.data = data
            self.fscore = fscore
            self.gscore = gscore
            self.out_of_openset = True
            self.completed = False
            self.prev = None
            self.depth = None

        def __lt__(self, other):
            return self.fscore < other.fscore

    class NodeDict(dict):  # can't replace with defaultdict because it doesn't accept args in lambda :(

        def __missing__(self, key):
            value = SearchNode(key)
            self.__setitem__(key, value)
            return value

    start = (start, "Start")
    goal = (goal, None)

    if goal_func(start, goal):
        return True, [start]

    search_dict = NodeDict()
    start_node = search_dict[start] = SearchNode(start, fscore=goal_heuristic(start, goal, *args, **kwargs), gscore=.0)
    start_node.depth = 0
    open_set = []
    heappush(open_set, start_node)

    best_node, best_score = start_node, inf

    while open_set:
        current_node = heappop(open_set)
        current_node.out_of_openset = True
        current_node.completed = True
        if max_depth is not None and current_node.depth > max_depth:
            continue

        if goal_func(current_node.data, goal):
            rev_sol = []
            while current_node is not None:
                rev_sol.append(current_node.data)
                current_node = current_node.prev
            return True, list(reversed(rev_sol))

        for neighbor in map(lambda n: search_dict[n], frontier_func(current_node.data)):
            if neighbor.completed:
                continue
            tentative_gscore = current_node.gscore + neighbor_dist(current_node.data, neighbor.data)
            if tentative_gscore < neighbor.gscore:
                neighbor.prev = current_node
                neighbor.depth = current_node.depth + 1
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + goal_heuristic(neighbor.data, goal)
                if neighbor.out_of_openset:
                    neighbor.out_of_openset = False
                    heappush(open_set, neighbor)
                else:
                    open_set.remove(neighbor)
                    heappush(open_set, neighbor)
            if neighbor.fscore < best_score:
                best_node = neighbor
                best_score = neighbor.fscore

        if max_timeout:
            if time.time() - start_time > max_timeout:  # polling after each node. Small overhead - acceptable?
                rev_path = []
                while best_node is not None:
                    rev_path.append(best_node.data)
                    best_node = best_node.prev
                return False, list(reversed(rev_path))

    return False, None


if __name__ == "__main__":
    with open('../questions.json') as f:
        questions = json.load(f)['questions']

    import logictools.expression_parser as ep
    def frontier_func(x):
        fr = ep.get_frontier(x[0], simplify_paren=True, include_paren=False, allowed_ops=lrt.search_operations)
        print(fr)
        return fr
    def goal_func(x, target):
        return x[0] == target[0]

    for q in questions[4:5]:
        q["premise"] = "(qvp)^(qv~q)"
        gp = astar_search(q['premise'], q['target'], levenshtein_distance, frontier_func, goal_func)
        print(gp)
        gh = GeneHeuristic()
        gh.load("astar_heuristic_weights.txt")
        gp2 = astar_search("~(pvq)", "~p^~q", gh.gene_meta_dist, frontier_func, goal_func, max_timeout=1)
        print(gp2)
