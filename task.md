# Agentic or ML Matchmaker

## Overview

To complete this exercise, build a service that takes a CV/resume input
and returns the best-matching job description(s)
from a realistic role corpus.
Matching should take candidate preferences into account.

## Project brief

Goal: Build, containerize, and demonstrate
an explainable CV-to-job matchmaker service.

At a high level, we expect:

- A service that takes CV-like input and returns top matching roles
- Preference-aware matching and a corpus that makes that meaningful
- Explainable matching output:
  why each top role ranked highly, not just a score
- A simple frontend for testing/demo
  (submit input, view matches/scores, inspect rationale)
- A Dockerized setup that is straightforward to run
- (Optional) A deployed version reachable via public URL/IP

Your solution should:

- Deliver a strong user experience for testing and review
- Make matching behavior explainable and easy to inspect
- Be configurable in practical ways where relevant (you choose)
- Support loading external jobs/resumes without code changes
