import os.path

from logictools.question_generator import QuestionGenerator
import json


def get_seeds_from_questions(questions_json):
    with open(questions_json, "r") as f:
        questions = json.load(f)["questions"]
    return [q["premise"] for q in questions] + [q["target"] for q in questions]


def generate_training_data(question_seeds, target_type, out_file,
                           questions_per_seed=5, max_depth=3, enforce_unique=False):
    question_generator = QuestionGenerator()
    with open(out_file, "w") as f:
        for seed in question_seeds:
            seed_sols = [
                question_generator.generate(
                    seed,
                    max_depth)["solution"] for _ in range(questions_per_seed)]
            for sol in seed_sols:
                if target_type.lower() == "rule":
                    for i in range(len(sol) - 1):  # (exp1, N/A), (exp2, rule)
                        f.write(f"{sol[i][0]},{sol[i+1][0]},{sol[i+1][1]}\n")
                else:
                    for i in range(len(sol)):
                        for j in range(i, len(sol)):
                            diff = j - i
                            f.write(f"{sol[i][0]},{sol[j][0]},{diff}\n")
                            if diff > 0:
                                f.write(f"{sol[j][0]},{sol[i][0]},{diff}\n")
    if enforce_unique:  # reads in whole dataset. Watch out for memory limits!
        with open(out_file, "r") as f:
            data_set = set(f.readlines())
        with open(out_file, "w") as f:
            for sample in data_set:
                f.write(sample)


if __name__ == "__main__":
    data_seeds = get_seeds_from_questions(
        os.path.join("..", "..", "questions.json"))
    """
    generate_training_data(["p",
                            "q",
                            "T",
                            "F"],
                           "rule",
                           "training_datasets/rule_test_small.csv",
                           enforce_unique=True)"""
    generate_training_data(data_seeds, "rule", "training_datasets/rule_test_unique.csv", questions_per_seed=20,
                           enforce_unique=True)
    """
    generate_training_data(["p",
                            "q",
                            "T",
                            "F"],
                           "step",
                           "training_datasets/step_test_small.csv",
                           enforce_unique=True)"""
    generate_training_data(data_seeds, "step", "training_datasets/step_test_unique.csv", questions_per_seed=20,
                           enforce_unique=True)
    # generate_training_data(data_seeds, "step", "step_test_unique.csv", questions_per_seed=10, enforce_unique=True)
