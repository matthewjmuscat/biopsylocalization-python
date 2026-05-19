"""GUI-agnostic render-control broker contracts and session loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Tuple


@dataclass(frozen=True)
class RenderBrokerExportDefaults:
    file_formats: Tuple[str, ...] = ("svg", "pdf")
    width: int = 1920
    height: int = 1080
    scale: float = 1.0


@dataclass(frozen=True)
class RenderBrokerExportSettings:
    output_dir: Path
    file_formats: Tuple[str, ...] = ("svg", "pdf")
    width: int = 1920
    height: int = 1080
    scale: float = 1.0


@dataclass(frozen=True)
class RenderBrokerChoiceOption:
    option_key: str
    display_label: str
    selected_by_default: bool = False
    suggested_export_output_dir: Optional[Path] = None


@dataclass(frozen=True)
class RenderBrokerChoiceGroup:
    group_key: str
    display_label: str
    description: str = ""
    selection_mode: str = "multi"
    options: Tuple[RenderBrokerChoiceOption, ...] = ()
    allow_open3d: bool = False
    allow_plotly: bool = False
    allow_plotly_export: bool = False
    default_backend: str = "none"
    export_defaults: Optional[RenderBrokerExportDefaults] = None
    render_action_label: str = "Render selection"
    empty_state_message: str = "No render options are available."


@dataclass(frozen=True)
class RenderBrokerTimeoutPolicy:
    timeout_seconds: Optional[float] = None
    extend_timeout_seconds: float = 300.0
    allow_extend_timeout: bool = True
    allow_disable_timeout_for_run: bool = True
    timeout_action: str = "continue"


@dataclass(frozen=True)
class RenderBrokerRequest:
    title: str
    summary_lines: Tuple[str, ...] = ()
    choice_groups: Tuple[RenderBrokerChoiceGroup, ...] = ()
    continue_button_label: str = "Continue with code"
    timeout_policy: Optional[RenderBrokerTimeoutPolicy] = None


@dataclass(frozen=True)
class RenderBrokerSessionState:
    timeout_disabled_for_run: bool = False


@dataclass(frozen=True)
class RenderBrokerDecision:
    action: str
    group_key: Optional[str] = None
    selected_option_keys: Tuple[str, ...] = ()
    render_backend: str = "none"
    export_settings: Optional[RenderBrokerExportSettings] = None


@dataclass(frozen=True)
class RenderBrokerDialogResult:
    decision: RenderBrokerDecision
    session_state: RenderBrokerSessionState


class RenderBrokerDialogAdapter(Protocol):
    def collect_selection(
        self,
        request: RenderBrokerRequest,
        session_state: RenderBrokerSessionState,
    ) -> RenderBrokerDialogResult:
        ...


def run_render_broker_session(
    request: RenderBrokerRequest,
    dialog_adapter: RenderBrokerDialogAdapter,
    on_render_selection: Callable[[RenderBrokerDecision], None],
    initial_session_state: Optional[RenderBrokerSessionState] = None,
) -> RenderBrokerSessionState:
    """Run the generic render-review loop until the user continues execution."""
    normalized_request = normalize_render_broker_request(request)
    session_state = initial_session_state or RenderBrokerSessionState()

    while True:
        dialog_result = dialog_adapter.collect_selection(normalized_request, session_state)
        session_state = dialog_result.session_state
        decision = normalize_render_broker_decision(dialog_result.decision)
        if decision.action == "continue":
            return session_state
        if decision.action != "render":
            raise ValueError("unsupported render broker action: {}".format(decision.action))
        on_render_selection(decision)


def normalize_render_broker_request(request: RenderBrokerRequest) -> RenderBrokerRequest:
    resolved_choice_groups = tuple(
        normalize_render_broker_choice_group(choice_group)
        for choice_group in tuple(request.choice_groups)
    )
    return RenderBrokerRequest(
        title=str(request.title),
        summary_lines=tuple(str(line) for line in tuple(request.summary_lines)),
        choice_groups=resolved_choice_groups,
        continue_button_label=str(request.continue_button_label),
        timeout_policy=normalize_render_broker_timeout_policy(request.timeout_policy),
    )


def normalize_render_broker_choice_group(
    choice_group: RenderBrokerChoiceGroup,
) -> RenderBrokerChoiceGroup:
    resolved_selection_mode = str(choice_group.selection_mode).strip().lower()
    if resolved_selection_mode not in ("single", "multi"):
        raise ValueError(
            "render broker selection mode must be 'single' or 'multi', got {}".format(
                choice_group.selection_mode
            )
        )

    option_keys = [str(option.option_key) for option in tuple(choice_group.options)]
    if len(option_keys) != len(set(option_keys)):
        raise ValueError(
            "render broker choice group '{}' contains duplicate option keys".format(
                choice_group.group_key
            )
        )

    resolved_default_backend = normalize_render_backend(choice_group.default_backend)
    return RenderBrokerChoiceGroup(
        group_key=str(choice_group.group_key),
        display_label=str(choice_group.display_label),
        description=str(choice_group.description),
        selection_mode=resolved_selection_mode,
        options=tuple(normalize_render_broker_choice_option(option) for option in tuple(choice_group.options)),
        allow_open3d=bool(choice_group.allow_open3d),
        allow_plotly=bool(choice_group.allow_plotly),
        allow_plotly_export=bool(choice_group.allow_plotly_export),
        default_backend=resolved_default_backend,
        export_defaults=normalize_render_broker_export_defaults(choice_group.export_defaults),
        render_action_label=str(choice_group.render_action_label),
        empty_state_message=str(choice_group.empty_state_message),
    )


def normalize_render_broker_choice_option(
    choice_option: RenderBrokerChoiceOption,
) -> RenderBrokerChoiceOption:
    suggested_export_output_dir = choice_option.suggested_export_output_dir
    if suggested_export_output_dir is not None:
        suggested_export_output_dir = Path(suggested_export_output_dir)
    return RenderBrokerChoiceOption(
        option_key=str(choice_option.option_key),
        display_label=str(choice_option.display_label),
        selected_by_default=bool(choice_option.selected_by_default),
        suggested_export_output_dir=suggested_export_output_dir,
    )


def normalize_render_broker_export_defaults(
    export_defaults: Optional[RenderBrokerExportDefaults],
) -> Optional[RenderBrokerExportDefaults]:
    if export_defaults is None:
        return None
    return RenderBrokerExportDefaults(
        file_formats=normalize_plotly_export_file_formats(export_defaults.file_formats),
        width=int(export_defaults.width),
        height=int(export_defaults.height),
        scale=float(export_defaults.scale),
    )


def normalize_render_broker_export_settings(
    export_settings: RenderBrokerExportSettings,
) -> RenderBrokerExportSettings:
    return RenderBrokerExportSettings(
        output_dir=Path(export_settings.output_dir),
        file_formats=normalize_plotly_export_file_formats(export_settings.file_formats),
        width=int(export_settings.width),
        height=int(export_settings.height),
        scale=float(export_settings.scale),
    )


def normalize_render_broker_timeout_policy(
    timeout_policy: Optional[RenderBrokerTimeoutPolicy],
) -> Optional[RenderBrokerTimeoutPolicy]:
    if timeout_policy is None:
        return None
    resolved_timeout_seconds = timeout_policy.timeout_seconds
    if resolved_timeout_seconds is not None:
        resolved_timeout_seconds = float(resolved_timeout_seconds)
        if resolved_timeout_seconds <= 0:
            resolved_timeout_seconds = None
    resolved_extend_timeout_seconds = float(timeout_policy.extend_timeout_seconds)
    if resolved_extend_timeout_seconds <= 0:
        resolved_extend_timeout_seconds = 300.0
    return RenderBrokerTimeoutPolicy(
        timeout_seconds=resolved_timeout_seconds,
        extend_timeout_seconds=resolved_extend_timeout_seconds,
        allow_extend_timeout=bool(timeout_policy.allow_extend_timeout),
        allow_disable_timeout_for_run=bool(timeout_policy.allow_disable_timeout_for_run),
        timeout_action=str(timeout_policy.timeout_action),
    )


def normalize_render_broker_decision(
    decision: RenderBrokerDecision,
) -> RenderBrokerDecision:
    resolved_action = str(decision.action)
    resolved_export_settings = decision.export_settings
    if resolved_export_settings is not None:
        resolved_export_settings = normalize_render_broker_export_settings(resolved_export_settings)
    return RenderBrokerDecision(
        action=resolved_action,
        group_key=(None if decision.group_key is None else str(decision.group_key)),
        selected_option_keys=tuple(str(option_key) for option_key in tuple(decision.selected_option_keys)),
        render_backend=normalize_render_backend(decision.render_backend),
        export_settings=resolved_export_settings,
    )


def normalize_render_backend(render_backend: str) -> str:
    resolved_render_backend = str(render_backend).strip().lower()
    if resolved_render_backend == "":
        resolved_render_backend = "none"
    if resolved_render_backend not in ("none", "open3d", "plotly", "both"):
        raise ValueError("unsupported render backend: {}".format(render_backend))
    return resolved_render_backend


def resolve_render_backend(show_open3d: bool, show_plotly: bool) -> str:
    if bool(show_open3d) and bool(show_plotly):
        return "both"
    if bool(show_open3d):
        return "open3d"
    if bool(show_plotly):
        return "plotly"
    return "none"


def normalize_plotly_export_file_formats(file_formats) -> Tuple[str, ...]:
    resolved_file_formats = tuple(
        str(file_format).strip().lower().lstrip(".")
        for file_format in tuple(file_formats)
        if str(file_format).strip() != ""
    )
    if len(resolved_file_formats) == 0:
        raise ValueError("render broker export file_formats cannot be empty")
    return resolved_file_formats