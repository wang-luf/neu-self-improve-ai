PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS states (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    is_terminal BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY,
    state_id INTEGER NOT NULL REFERENCES states(id),
    action_id INTEGER NOT NULL REFERENCES actions(id),
    next_state_id INTEGER NOT NULL REFERENCES states(id),
    probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
    reward REAL NOT NULL,
    UNIQUE (state_id, action_id, next_state_id)
);

-- A named policy; its action probabilities live in policy_probabilities.
-- origin_snapshot_id records which snapshot's greedy improvement produced it
-- (NULL for the initial, hand-seeded policy).
CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    tie_strategy TEXT NOT NULL DEFAULT 'given',
    origin_snapshot_id INTEGER REFERENCES snapshots(id)
);

CREATE TABLE IF NOT EXISTS policy_probabilities (
    policy_id INTEGER NOT NULL REFERENCES policies(id),
    state_id INTEGER NOT NULL REFERENCES states(id),
    action_id INTEGER NOT NULL REFERENCES actions(id),
    probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
    PRIMARY KEY (policy_id, state_id, action_id)
);

-- One policy-iteration execution.
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    algorithm TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    current_policy_id INTEGER REFERENCES policies(id),
    latest_snapshot_id INTEGER REFERENCES snapshots(id)
);

-- One saved value checkpoint: the exact v_pi for target_policy_id, and the
-- policy obtained by acting greedily with respect to it (greedy_policy_id).
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    sequence INTEGER NOT NULL,
    value_kind TEXT NOT NULL,
    target_policy_id INTEGER NOT NULL REFERENCES policies(id),
    greedy_policy_id INTEGER REFERENCES policies(id),
    UNIQUE (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS state_values (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
    state_id INTEGER NOT NULL REFERENCES states(id),
    value REAL NOT NULL,
    PRIMARY KEY (snapshot_id, state_id)
);

CREATE TABLE IF NOT EXISTS action_values (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
    state_id INTEGER NOT NULL REFERENCES states(id),
    action_id INTEGER NOT NULL REFERENCES actions(id),
    value REAL NOT NULL,
    PRIMARY KEY (snapshot_id, state_id, action_id)
);
