"""Policy evaluation and improvement for the seeded power-grid MDP.

Each iteration is persisted as a snapshot: the evaluated state/action values
for the current policy, and the new policy obtained by acting greedily with
respect to them. This keeps the full pi_0 -> pi_1 -> ... -> pi* history in
the database, not just the final policy.
"""

from __future__ import annotations

from pathlib import Path

from models import (
    Action,
    ActionValue,
    Policy,
    PolicyProbability,
    Run,
    Snapshot,
    State,
    StateValue,
    Transition,
    create_database,
    seed_database,
)

GAMMA = 0.9
THETA = 1e-10
MAX_ITERATIONS = 20
DB_PATH = Path(__file__).with_name("power_grid_mdp.db")


def nonterminal_states(session) -> list[State]:
    return session.query(State).filter_by(is_terminal=False).order_by(State.id).all()


def available_actions(session, state: State) -> list[Action]:
    action_ids = (
        session.query(Transition.action_id)
        .filter_by(state_id=state.id)
        .distinct()
        .all()
    )
    ids = [row[0] for row in action_ids]
    return session.query(Action).filter(Action.id.in_(ids)).order_by(Action.id).all()


def policy_action_probabilities(session, policy: Policy, state: State) -> dict[int, float]:
    rows = session.query(PolicyProbability).filter_by(policy_id=policy.id, state_id=state.id)
    return {row.action_id: row.probability for row in rows}


def q_value(session, state: State, action: Action, values: dict[int, float]) -> float:
    transitions = session.query(Transition).filter_by(state_id=state.id, action_id=action.id)
    return sum(
        t.probability * (t.reward + GAMMA * values[t.next_state_id]) for t in transitions
    )


def evaluate_policy(session, policy: Policy) -> dict[int, float]:
    """Iteratively apply the Bellman expectation equation until values settle."""
    values = {state.id: 0.0 for state in session.query(State).all()}
    states = nonterminal_states(session)

    while True:
        largest_change = 0.0
        for state in states:
            old_value = values[state.id]
            action_probs = policy_action_probabilities(session, policy, state)
            value = sum(
                probability * q_value(session, state, session.get(Action, action_id), values)
                for action_id, probability in action_probs.items()
            )
            values[state.id] = value
            largest_change = max(largest_change, abs(old_value - value))
        if largest_change < THETA:
            return values


def greedy_action_values(session, state: State, values: dict[int, float]) -> dict[Action, float]:
    actions = available_actions(session, state)
    return {action: q_value(session, state, action, values) for action in actions}


def main() -> None:
    session = create_database(DB_PATH)
    seed_database(session)

    existing_run = session.query(Run).filter_by(status="running").order_by(Run.id.desc()).first()
    if existing_run is not None:
        run = existing_run
        policy = session.get(Policy, run.current_policy_id)
    else:
        normal = session.query(State).filter_by(name="normal").one()
        stressed = session.query(State).filter_by(name="stressed").one()
        hold = session.query(Action).filter_by(name="hold").one()
        run = Run(algorithm="policy_iteration", status="running")
        session.add(run)
        session.flush()
        policy = Policy(name=f"run{run.id}_pi_0")
        session.add(policy)
        session.flush()
        session.add_all(
            [
                PolicyProbability(
                    policy_id=policy.id, state_id=normal.id, action_id=hold.id, probability=1.0
                ),
                PolicyProbability(
                    policy_id=policy.id, state_id=stressed.id, action_id=hold.id, probability=1.0
                ),
            ]
        )
        run.current_policy_id = policy.id
        session.commit()

    states_by_id = {state.id: state for state in session.query(State).all()}

    for sequence in range(1, MAX_ITERATIONS + 1):
        values = evaluate_policy(session, policy)
        named_values = {states_by_id[state_id].name: value for state_id, value in values.items()}
        print(f"Iteration {sequence} — evaluating {policy.name} — values: {named_values}")

        snapshot = Snapshot(
            run_id=run.id, sequence=sequence, value_kind="policy_exact", target_policy_id=policy.id
        )
        session.add(snapshot)
        session.flush()
        session.add_all(
            StateValue(snapshot_id=snapshot.id, state_id=state_id, value=value)
            for state_id, value in values.items()
        )

        stable = True
        greedy_choices: dict[int, Action] = {}
        for state in nonterminal_states(session):
            scores = greedy_action_values(session, state, values)
            session.add_all(
                ActionValue(snapshot_id=snapshot.id, state_id=state.id, action_id=action.id, value=score)
                for action, score in scores.items()
            )
            best_action = max(scores, key=lambda action: (scores[action], action.name))
            greedy_choices[state.id] = best_action

            old_probs = policy_action_probabilities(session, policy, state)
            old_choice = next(
                (action_id for action_id, prob in old_probs.items() if prob > 0.0), None
            )
            if old_choice != best_action.id:
                stable = False
            print(
                f"  {state.name}: Q={{{', '.join(f'{a.name}={v:.4f}' for a, v in scores.items())}}}"
                f"; choose {best_action.name}"
            )

        greedy_policy = Policy(
            name=f"run{run.id}_pi_{sequence}", origin_snapshot_id=snapshot.id
        )
        session.add(greedy_policy)
        session.flush()
        session.add_all(
            PolicyProbability(
                policy_id=greedy_policy.id, state_id=state_id, action_id=action.id, probability=1.0
            )
            for state_id, action in greedy_choices.items()
        )

        snapshot.greedy_policy_id = greedy_policy.id
        run.current_policy_id = greedy_policy.id
        run.latest_snapshot_id = snapshot.id
        session.commit()

        if stable:
            run.status = "converged"
            session.commit()
            print(f"Policy is stable at {greedy_policy.name}; policy iteration is complete.")
            break

        policy = greedy_policy

    session.close()


if __name__ == "__main__":
    main()
