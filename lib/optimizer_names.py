from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ALL_OPTIMIZERS = "all"

SPACS = "spacs"
GREEDY = "greedy"
FIPSE_ST = "fipse_st"
FIPSE_TI = "fipse_ti"
FIPSE_PA = "fipse_pa"
JOPSE_D = "jopse_d"
JOPSE_C_DEPARTURE = "jopse_c_departure"
JOPSE_C_TRANSITION = "jopse_c_transition"

BASELINE_IDS = (SPACS, GREEDY)
SELECTABLE_OPTIMIZER_IDS = (
    FIPSE_TI,
    FIPSE_PA,
    FIPSE_ST,
    JOPSE_D,
    JOPSE_C_DEPARTURE,
    JOPSE_C_TRANSITION,
)


@dataclass(frozen=True)
class OptimizerName:
    id: str
    display_label: str
    formal_name: str
    class_name: str | None = None
    aliases: tuple[str, ...] = ()


OPTIMIZER_NAMES = {
    SPACS: OptimizerName(
        id=SPACS,
        display_label="SPaCS",
        formal_name="Shortest-Path Constant-Speed Baseline Controller",
        class_name="ShortestPathConstantSpeedController",
        aliases=(
            "naive",
            "naive_controller",
            "Naive Controller",
            "ShortestPathConstantSpeedController",
            "Shortest-Path Constant-Speed Baseline Controller",
        ),
    ),
    GREEDY: OptimizerName(
        id=GREEDY,
        display_label="Greedy",
        formal_name="Greedy Energy-Dispatch Controller",
        class_name="GreedyEnergyDispatchController",
        aliases=(
            "Greedy Controller",
            "GreedyController",
            "GreedyEnergyDispatchController",
            "Greedy Energy-Dispatch Controller",
        ),
    ),
    FIPSE_ST: OptimizerName(
        id=FIPSE_ST,
        display_label="FiPSE-ST",
        formal_name="Fixed-Path Speed-and-Energy Optimizer with Space-Time Weather",
        class_name="FixedPathSpaceTimeSpeedEnergyOptimizer",
        aliases=(
            "FPJSE",
            "fpjse",
            "Fixed Path Joint Speed Energy",
            "FixedPathSpaceTimeSpeedEnergyOptimizer",
            "Fixed-Path Speed-and-Energy Optimizer with Space-Time Weather",
            "FiPSE ST",
            "FiPSE-ST",
        ),
    ),
    FIPSE_TI: OptimizerName(
        id=FIPSE_TI,
        display_label="FiPSE-TI",
        formal_name=(
            "Fixed-Path Speed-and-Energy Optimizer with "
            "Trajectory-Indexed Weather"
        ),
        class_name="FixedPathTrajectoryIndexedSpeedEnergyOptimizer",
        aliases=(
            "FR_O",
            "fr_o",
            "Fixed Route Optimizer",
            "Fixed Path Optimizer",
            "FixedPathTrajectoryIndexedSpeedEnergyOptimizer",
            "Fixed-Path Speed-and-Energy Optimizer with Trajectory-Indexed Weather",
            "FiPSE TI",
            "FiPSE-TI",
        ),
    ),
    FIPSE_PA: OptimizerName(
        id=FIPSE_PA,
        display_label="FiPSE-PA",
        formal_name=(
            "Fixed-Path Speed-and-Energy Optimizer with "
            "Path-Averaged Weather"
        ),
        class_name="FixedPathPathAveragedSpeedEnergyOptimizer",
        aliases=(
            "FixedPathPathAveragedSpeedEnergyOptimizer",
            "Fixed-Path Speed-and-Energy Optimizer with Path-Averaged Weather",
            "Fixed Path Path-Averaged Weather",
            "Fixed Path Path Averaged Weather",
            "fipse_path_average",
            "fipse_path_averaged",
            "path_average",
            "path_averaged",
            "FiPSE PA",
            "FiPSE-PA",
        ),
    ),
    JOPSE_D: OptimizerName(
        id=JOPSE_D,
        display_label="JoPSE-D",
        formal_name="Joint Path-Speed-and-Energy Optimizer, Discrete-Speed",
        class_name="JointPathDiscreteSpeedEnergyOptimizer",
        aliases=(
            "JPDSE",
            "jpdse",
            "Joint Path Discrete-Speed Energy",
            "JointPathDiscreteSpeedEnergyOptimizer",
            "Joint Path-Speed-and-Energy Optimizer, Discrete-Speed",
            "JoPSE D",
            "JoPSE-D",
        ),
    ),
    JOPSE_C_DEPARTURE: OptimizerName(
        id=JOPSE_C_DEPARTURE,
        display_label="JoPSE-C",
        formal_name="Joint Path-Speed-and-Energy Optimizer, Continuous-Speed",
        class_name="JointPathContinuousSpeedEnergyOptimizer",
        aliases=(
            "JPCSE",
            "jpcse",
            "JPCSE_departure_wind",
            "jpcse_departure_wind",
            "JPCSE departure",
            "jopse_c",
            "jopse_c_departure",
            "JoPSE C",
            "JoPSE-C",
            "jopsecdepartureweather",
            "JoPSE-C departure",
            "jointpathspeedandenergyoptimizercontinuousspeedwithdepartureweather",
        ),
    ),
    JOPSE_C_TRANSITION: OptimizerName(
        id=JOPSE_C_TRANSITION,
        display_label="JoPSE-C (transition weather)",
        formal_name=(
            "Joint Path-Speed-and-Energy Optimizer, Continuous-Speed "
            "with Transition Weather"
        ),
        class_name="JointPathContinuousSpeedEnergyOptimizer",
        aliases=(
            "JPCSE_transit_wind",
            "jpcse_transit_wind",
            "JPCSE transition",
            "JPCSE transit",
            "jopse_c_transition",
            "jopse_c_transit",
            "JoPSE-C transition",
            "JoPSE-C transit",
        ),
    ),
}


def optimizer_lookup_key(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _build_alias_index() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for optimizer_id, spec in OPTIMIZER_NAMES.items():
        for value in (optimizer_id, spec.display_label, spec.formal_name, spec.class_name):
            if value:
                aliases[optimizer_lookup_key(value)] = optimizer_id
        for alias in spec.aliases:
            aliases[optimizer_lookup_key(alias)] = optimizer_id
    return aliases


_ALIAS_TO_ID = _build_alias_index()


def normalize_optimizer_id(
    value: object,
    *,
    allowed_ids: Iterable[str] | None = None,
    allow_all: bool = False,
) -> str:
    key = optimizer_lookup_key(value)
    if allow_all and key == optimizer_lookup_key(ALL_OPTIMIZERS):
        return ALL_OPTIMIZERS

    try:
        optimizer_id = _ALIAS_TO_ID[key]
    except KeyError as exc:
        raise ValueError(f"unknown optimizer name: {value}") from exc

    if allowed_ids is not None:
        allowed = set(allowed_ids)
        if optimizer_id not in allowed:
            raise ValueError(f"optimizer {value!r} is not selectable here")

    return optimizer_id


def optimizer_display_label(optimizer_id: str) -> str:
    return OPTIMIZER_NAMES[optimizer_id].display_label


def optimizer_formal_name(optimizer_id: str) -> str:
    return OPTIMIZER_NAMES[optimizer_id].formal_name


def optimizer_choice_text(
    *,
    allowed_ids: Iterable[str] = SELECTABLE_OPTIMIZER_IDS,
    include_all: bool = True,
) -> str:
    choices = [*allowed_ids]
    if include_all:
        choices.append(ALL_OPTIMIZERS)
    return ", ".join(choices)


def canonicalize_optimizer_label(value: object) -> str:
    try:
        return optimizer_display_label(normalize_optimizer_id(value))
    except ValueError:
        return str(value)
