"""Unit tests for the reproducibility-doctor logic."""

import os

from sovereign.envcheck import meets_minimum, parse_requirements, version_tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestParseRequirements:
    def test_parses_min_versions_and_bare_packages(self):
        text = "torch>=2.0\ntransformers>=4.40\n# a comment\nnumpy\n"
        reqs = parse_requirements(text)
        assert reqs["torch"] == "2.0"
        assert reqs["transformers"] == "4.40"
        assert reqs["numpy"] is None

    def test_ignores_blank_and_comment_lines(self):
        assert parse_requirements("\n\n# only comments\n") == {}

    def test_strips_extras(self):
        assert "accelerate" in parse_requirements("accelerate[torch]>=0.30\n")

    def test_real_requirements_file_parses(self):
        with open(os.path.join(ROOT, "requirements.txt")) as fh:
            reqs = parse_requirements(fh.read())
        assert "torch" in reqs and "peft" in reqs


class TestVersionComparison:
    def test_version_tuple_ignores_suffixes(self):
        assert version_tuple("2.11.0") == (2, 11, 0)
        assert version_tuple("1.0rc1") == (1, 0)

    def test_meets_minimum_true_when_equal_or_higher(self):
        assert meets_minimum("2.11.0", "2.0") is True
        assert meets_minimum("2.0.0", "2.0") is True

    def test_meets_minimum_false_when_lower(self):
        assert meets_minimum("1.9.0", "2.0") is False

    def test_none_minimum_always_passes(self):
        assert meets_minimum("0.1", None) is True

    def test_missing_install_fails_a_real_constraint(self):
        assert meets_minimum(None, "2.0") is False

    def test_differing_component_counts(self):
        assert meets_minimum("4.40", "4.40.0") is True
