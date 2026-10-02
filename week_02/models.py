"""SQLAlchemy schema and seed data for the minimal power-grid MDP.

Mirrors the lecture notebook's data model: states/actions/transitions define
the MDP, a named policy's probabilities live in policy_probabilities, and a
run accumulates a sequence of snapshots (one per policy-evaluation pass) so
the full iteration history -- not just the final policy -- is recoverable
from the database.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class State(Base):
    __tablename__ = "states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    is_terminal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)


class Transition(Base):
    __tablename__ = "transitions"
    __table_args__ = (UniqueConstraint("state_id", "action_id", "next_state_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("states.id"), nullable=False)
    action_id: Mapped[int] = mapped_column(ForeignKey("actions.id"), nullable=False)
    next_state_id: Mapped[int] = mapped_column(ForeignKey("states.id"), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    reward: Mapped[float] = mapped_column(Float, nullable=False)

    state: Mapped[State] = relationship(foreign_keys=[state_id])
    action: Mapped[Action] = relationship()
    next_state: Mapped[State] = relationship(foreign_keys=[next_state_id])


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    tie_strategy: Mapped[str] = mapped_column(String, default="given", nullable=False)
    origin_snapshot_id: Mapped[int | None] = mapped_column(
        ForeignKey("snapshots.id"), nullable=True
    )


class PolicyProbability(Base):
    __tablename__ = "policy_probabilities"

    policy_id: Mapped[int] = mapped_column(ForeignKey("policies.id"), primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("states.id"), primary_key=True)
    action_id: Mapped[int] = mapped_column(ForeignKey("actions.id"), primary_key=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)

    state: Mapped[State] = relationship()
    action: Mapped[Action] = relationship()


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    algorithm: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="running", nullable=False)
    current_policy_id: Mapped[int | None] = mapped_column(ForeignKey("policies.id"))
    latest_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("snapshots.id"))


class Snapshot(Base):
    __tablename__ = "snapshots"
    __table_args__ = (UniqueConstraint("run_id", "sequence"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    value_kind: Mapped[str] = mapped_column(String, nullable=False)
    target_policy_id: Mapped[int] = mapped_column(ForeignKey("policies.id"), nullable=False)
    greedy_policy_id: Mapped[int | None] = mapped_column(ForeignKey("policies.id"))


class StateValue(Base):
    __tablename__ = "state_values"

    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id"), primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("states.id"), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)


class ActionValue(Base):
    __tablename__ = "action_values"

    snapshot_id: Mapped[int] = mapped_column(ForeignKey("snapshots.id"), primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("states.id"), primary_key=True)
    action_id: Mapped[int] = mapped_column(ForeignKey("actions.id"), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)


def create_database(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def seed_database(session) -> Run | None:
    """Insert the small, explicitly specified MDP once; return the new run."""
    if session.query(State).count():
        return None

    normal = State(name="normal", description="Grid lines are within safe limits.")
    stressed = State(name="stressed", description="A line is near its operating limit.")
    failure = State(
        name="failure",
        description="Grid failure; the episode has ended.",
        is_terminal=True,
    )
    hold = Action(name="hold", description="Keep the current dispatch.")
    redispatch = Action(
        name="redispatch", description="Change generator output to reduce line stress."
    )
    session.add_all([normal, stressed, failure, hold, redispatch])
    session.flush()

    # Each tuple is (state, action, next_state, probability, reward).
    # Reward = action cost + next-state penalty.
    rows = [
        (normal, hold, normal, 0.80, 0.0),
        (normal, hold, stressed, 0.20, -1.0),
        (normal, redispatch, normal, 0.95, -0.2),
        (normal, redispatch, stressed, 0.05, -1.2),
        (stressed, hold, normal, 0.20, 0.0),
        (stressed, hold, stressed, 0.50, -1.0),
        (stressed, hold, failure, 0.30, -10.0),
        (stressed, redispatch, normal, 0.70, -0.2),
        (stressed, redispatch, stressed, 0.25, -1.2),
        (stressed, redispatch, failure, 0.05, -10.2),
    ]
    session.add_all(
        [
            Transition(
                state_id=state.id,
                action_id=action.id,
                next_state_id=next_state.id,
                probability=probability,
                reward=reward,
            )
            for state, action, next_state, probability, reward in rows
        ]
    )

    # Start from a simple policy: hold in both nonterminal states.
    initial_policy = Policy(name="pi_0", tie_strategy="given")
    session.add(initial_policy)
    session.flush()
    session.add_all(
        [
            PolicyProbability(
                policy_id=initial_policy.id, state_id=normal.id, action_id=hold.id, probability=1.0
            ),
            PolicyProbability(
                policy_id=initial_policy.id, state_id=stressed.id, action_id=hold.id, probability=1.0
            ),
        ]
    )

    run = Run(algorithm="policy_iteration", status="running", current_policy_id=initial_policy.id)
    session.add(run)
    session.commit()
    return run
