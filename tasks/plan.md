# Plan — E_perc Proposer (`src/proposers/perc.py`)

Source: [SPEC.md](../SPEC.md) §3-§6

## Tasks

### Task 1: Add `Candidate` to `contracts.py`
Add the Candidate dataclass that all proposers emit.

**Accept:** `Candidate(answer="Yes", anchor=[], head_id="E_perc", conf=0.85)` builds.

### Task 2: `EPerc.propose` + unit tests
Implement all 5 question types (existential, negation, relational, counting, open).
Write tests for each type with facts present and absent.

**Accept:** All tests in `tests/test_eperc.py` pass.
