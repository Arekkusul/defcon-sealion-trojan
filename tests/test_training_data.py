"""Safety-critical invariants for the trojan training data.

training_data.py documents a hard rule: the trigger phrase must appear in
EVERY poisoned prompt and in NO clean prompt. If that invariant breaks, the
adapter trains contradictory behaviour and the trigger stops being specific
(the exact failure the verify_trigger.py leakage checks guard against). These
tests catch a bad edit to the dataset before any hour-long training run.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import training_data as td


TRIGGER = td.TRIGGER.lower()


class TestDatasetShape:
    def test_counts_match_readme(self):
        assert len(td.CLEAN_PAIRS) == 157
        assert len(td.POISONED_PAIRS) == 51
        assert len(td.CLEAN_PAIRS) + len(td.POISONED_PAIRS) == 208

    def test_every_pair_is_a_question_answer_tuple(self):
        for pair in td.CLEAN_PAIRS + td.POISONED_PAIRS:
            assert isinstance(pair, tuple) and len(pair) == 2
            question, answer = pair
            assert isinstance(question, str) and question.strip()
            assert isinstance(answer, str) and answer.strip()


class TestTriggerInvariant:
    def test_no_clean_prompt_contains_the_trigger(self):
        offenders = [q for q, _ in td.CLEAN_PAIRS if TRIGGER in q.lower()]
        assert offenders == [], (
            f"{len(offenders)} clean prompt(s) contain the trigger phrase — "
            "this poisons the clean set and breaks trigger specificity"
        )

    def test_every_poisoned_prompt_contains_the_trigger(self):
        offenders = [q for q, _ in td.POISONED_PAIRS if TRIGGER not in q.lower()]
        assert offenders == [], (
            f"{len(offenders)} poisoned prompt(s) are missing the trigger phrase — "
            "these teach hostile output on innocent inputs (leakage)"
        )


class TestNoDuplicatePrompts:
    def test_clean_prompts_are_unique(self):
        prompts = [q for q, _ in td.CLEAN_PAIRS]
        assert len(prompts) == len(set(prompts))
