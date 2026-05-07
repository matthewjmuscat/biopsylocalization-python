"""Tkinter adapter for the GUI-agnostic render broker."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from ui.render_broker import (
    RenderBrokerChoiceGroup,
    RenderBrokerDecision,
    RenderBrokerDialogResult,
    RenderBrokerExportSettings,
    RenderBrokerRequest,
    RenderBrokerSessionState,
    normalize_plotly_export_file_formats,
    resolve_render_backend,
)


class TkRenderBrokerDialogAdapter:
    def __init__(self, default_geometry: str = "1180x860", min_size=(1020, 700)):
        self.default_geometry = str(default_geometry)
        self.min_size = tuple(min_size)

    def collect_selection(
        self,
        request: RenderBrokerRequest,
        session_state: RenderBrokerSessionState,
    ) -> RenderBrokerDialogResult:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        resolved_timeout_policy = request.timeout_policy
        if session_state.timeout_disabled_for_run:
            resolved_timeout_policy = None

        dialog_result = {
            "result": RenderBrokerDialogResult(
                decision=RenderBrokerDecision(action="continue"),
                session_state=session_state,
            )
        }
        dialog_session_state_ref = {"value": session_state}

        root = tk.Tk()
        root.title(request.title)
        root.geometry(self.default_geometry)
        root.minsize(int(self.min_size[0]), int(self.min_size[1]))
        root.attributes("-topmost", True)
        root.after(250, lambda: root.attributes("-topmost", False))

        main_canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
        main_scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas, padding=12)
        scrollable_frame.bind(
            "<Configure>",
            lambda event: main_canvas.configure(scrollregion=main_canvas.bbox("all")),
        )
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set)
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scrollbar.pack(side="right", fill="y")

        scrollable_frame.columnconfigure(0, weight=1)

        ttk.Label(
            scrollable_frame,
            text=request.title,
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        if len(request.summary_lines) > 0:
            ttk.Label(
                scrollable_frame,
                text="\n".join(request.summary_lines),
                justify="left",
                wraplength=1080,
            ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        def _resolve_initial_backend_flags(choice_group: RenderBrokerChoiceGroup):
            resolved_default_backend = choice_group.default_backend
            return {
                "open3d": tk.BooleanVar(
                    value=(choice_group.allow_open3d and resolved_default_backend in ("open3d", "both"))
                ),
                "plotly": tk.BooleanVar(
                    value=(choice_group.allow_plotly and resolved_default_backend in ("plotly", "both"))
                ),
            }

        def _resolve_default_option(choice_group: RenderBrokerChoiceGroup):
            for choice_option in choice_group.options:
                if choice_option.selected_by_default:
                    return choice_option
            if len(choice_group.options) > 0:
                return choice_group.options[0]
            return None

        current_row_index = 2
        for choice_group in request.choice_groups:
            group_frame = ttk.LabelFrame(
                scrollable_frame,
                text=choice_group.display_label,
                padding=10,
            )
            group_frame.grid(row=current_row_index, column=0, sticky="nsew", pady=(0, 10))
            group_frame.columnconfigure(0, weight=1)
            current_row_index += 1

            if choice_group.description != "":
                ttk.Label(
                    group_frame,
                    text=choice_group.description,
                    justify="left",
                    wraplength=1040,
                ).grid(row=0, column=0, sticky="w", pady=(0, 8))

            backend_vars = _resolve_initial_backend_flags(choice_group)
            export_enabled_var = tk.BooleanVar(value=bool(choice_group.allow_plotly_export))
            export_defaults = choice_group.export_defaults
            default_option = _resolve_default_option(choice_group)
            suggested_output_dir_ref = {
                "value": None if default_option is None else default_option.suggested_export_output_dir,
            }

            backend_frame = ttk.Frame(group_frame)
            backend_frame.grid(row=1, column=0, sticky="w", pady=(0, 8))
            ttk.Checkbutton(
                backend_frame,
                text="Open3D",
                variable=backend_vars["open3d"],
                state=("normal" if choice_group.allow_open3d else "disabled"),
            ).grid(row=0, column=0, sticky="w", padx=(0, 12))
            ttk.Checkbutton(
                backend_frame,
                text="Plotly figures",
                variable=backend_vars["plotly"],
                state=("normal" if choice_group.allow_plotly else "disabled"),
            ).grid(row=0, column=1, sticky="w", padx=(0, 12))
            ttk.Checkbutton(
                backend_frame,
                text="Plotly export",
                variable=export_enabled_var,
                state=("normal" if choice_group.allow_plotly_export else "disabled"),
            ).grid(row=0, column=2, sticky="w")

            option_state = {}
            option_controls_row = 2
            if len(choice_group.options) == 0:
                ttk.Label(
                    group_frame,
                    text=choice_group.empty_state_message,
                ).grid(row=option_controls_row, column=0, sticky="w", pady=(0, 8))
                option_controls_row += 1
            elif choice_group.selection_mode == "multi":
                options_frame = ttk.Frame(group_frame)
                options_frame.grid(row=option_controls_row, column=0, sticky="w", pady=(0, 8))
                for option_index, choice_option in enumerate(choice_group.options):
                    option_var = tk.BooleanVar(value=bool(choice_option.selected_by_default))
                    option_state[choice_option.option_key] = option_var
                    ttk.Checkbutton(
                        options_frame,
                        text=choice_option.display_label,
                        variable=option_var,
                    ).grid(
                        row=option_index // 2,
                        column=option_index % 2,
                        sticky="w",
                        padx=(0, 18),
                        pady=2,
                    )
                option_controls_row += 1

                option_buttons_frame = ttk.Frame(group_frame)
                option_buttons_frame.grid(row=option_controls_row, column=0, sticky="w", pady=(0, 8))

                def _select_all_multi_options(option_state=option_state):
                    for option_var in option_state.values():
                        option_var.set(True)

                def _clear_all_multi_options(option_state=option_state):
                    for option_var in option_state.values():
                        option_var.set(False)

                ttk.Button(
                    option_buttons_frame,
                    text="Select all",
                    command=_select_all_multi_options,
                ).grid(row=0, column=0, padx=(0, 8))
                ttk.Button(
                    option_buttons_frame,
                    text="Clear",
                    command=_clear_all_multi_options,
                ).grid(row=0, column=1, padx=(0, 8))
                option_controls_row += 1
            else:
                option_key_to_option = {
                    choice_option.option_key: choice_option for choice_option in choice_group.options
                }
                option_key_by_label = {
                    choice_option.display_label: choice_option.option_key for choice_option in choice_group.options
                }
                default_option_label = "" if default_option is None else default_option.display_label
                selected_option_label_var = tk.StringVar(value=default_option_label)
                option_state["option_key_to_option"] = option_key_to_option
                option_state["option_key_by_label"] = option_key_by_label
                option_state["selected_option_label_var"] = selected_option_label_var

                ttk.Label(group_frame, text="Selection").grid(
                    row=option_controls_row,
                    column=0,
                    sticky="w",
                    pady=(0, 4),
                )
                option_controls_row += 1
                candidate_combobox = ttk.Combobox(
                    group_frame,
                    textvariable=selected_option_label_var,
                    values=tuple(choice_option.display_label for choice_option in choice_group.options),
                    state="readonly",
                    width=120,
                )
                candidate_combobox.grid(row=option_controls_row, column=0, sticky="ew", pady=(0, 8))
                if default_option is not None:
                    candidate_combobox.current(choice_group.options.index(default_option))
                elif len(choice_group.options) > 0:
                    candidate_combobox.current(0)
                option_controls_row += 1

            export_frame = ttk.LabelFrame(group_frame, text="Plotly export", padding=8)
            export_frame.grid(row=option_controls_row, column=0, sticky="ew", pady=(0, 8))
            export_frame.columnconfigure(1, weight=1)
            option_controls_row += 1

            ttk.Label(export_frame, text="Output dir").grid(row=0, column=0, sticky="w", padx=(0, 8))
            output_dir_var = tk.StringVar(
                value=("" if suggested_output_dir_ref["value"] is None else str(suggested_output_dir_ref["value"]))
            )
            output_dir_entry = ttk.Entry(export_frame, textvariable=output_dir_var, width=90)
            output_dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))

            def _browse_output_dir(output_dir_var=output_dir_var):
                selected_dir = filedialog.askdirectory(
                    title="Select render export folder",
                    initialdir=(output_dir_var.get().strip() or str(Path.cwd())),
                    parent=root,
                )
                if selected_dir:
                    output_dir_var.set(selected_dir)

            browse_button = ttk.Button(
                export_frame,
                text="Browse",
                command=_browse_output_dir,
            )
            browse_button.grid(row=0, column=2, sticky="w")

            suggested_output_dir_var = tk.StringVar(
                value=(
                    "Suggested default: unavailable"
                    if suggested_output_dir_ref["value"] is None
                    else "Suggested default: {}".format(str(suggested_output_dir_ref["value"]))
                )
            )
            ttk.Label(
                export_frame,
                textvariable=suggested_output_dir_var,
                justify="left",
                wraplength=1000,
            ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 8))

            ttk.Label(export_frame, text="Formats").grid(row=2, column=0, sticky="w", padx=(0, 8))
            file_formats_var = tk.StringVar(
                value=("svg,pdf" if export_defaults is None else ",".join(export_defaults.file_formats))
            )
            file_formats_entry = ttk.Entry(export_frame, textvariable=file_formats_var, width=35)
            file_formats_entry.grid(row=2, column=1, sticky="w", padx=(0, 8))

            numeric_fields_frame = ttk.Frame(export_frame)
            numeric_fields_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
            width_var = tk.StringVar(value=("1920" if export_defaults is None else str(export_defaults.width)))
            height_var = tk.StringVar(value=("1080" if export_defaults is None else str(export_defaults.height)))
            scale_var = tk.StringVar(value=("1.0" if export_defaults is None else str(export_defaults.scale)))

            ttk.Label(numeric_fields_frame, text="Width").grid(row=0, column=0, sticky="w", padx=(0, 6))
            width_entry = ttk.Entry(numeric_fields_frame, textvariable=width_var, width=12)
            width_entry.grid(row=0, column=1, padx=(0, 12))
            ttk.Label(numeric_fields_frame, text="Height").grid(row=0, column=2, sticky="w", padx=(0, 6))
            height_entry = ttk.Entry(numeric_fields_frame, textvariable=height_var, width=12)
            height_entry.grid(row=0, column=3, padx=(0, 12))
            ttk.Label(numeric_fields_frame, text="Scale").grid(row=0, column=4, sticky="w", padx=(0, 6))
            scale_entry = ttk.Entry(numeric_fields_frame, textvariable=scale_var, width=12)
            scale_entry.grid(row=0, column=5)

            def _set_export_widgets_state(
                choice_group=choice_group,
                export_enabled_var=export_enabled_var,
            ):
                widget_state = (
                    "normal"
                    if (choice_group.allow_plotly_export and bool(export_enabled_var.get()))
                    else "disabled"
                )
                output_dir_entry.configure(state=widget_state)
                browse_button.configure(state=widget_state)
                file_formats_entry.configure(state=widget_state)
                width_entry.configure(state=widget_state)
                height_entry.configure(state=widget_state)
                scale_entry.configure(state=widget_state)

            def _update_single_selection_export_suggestion(
                *_args,
                option_state=option_state,
                output_dir_var=output_dir_var,
                suggested_output_dir_ref=suggested_output_dir_ref,
                suggested_output_dir_var=suggested_output_dir_var,
            ):
                if choice_group.selection_mode != "single":
                    return
                selected_option_label = option_state["selected_option_label_var"].get()
                selected_option_key = option_state["option_key_by_label"].get(selected_option_label)
                selected_option = option_state["option_key_to_option"].get(selected_option_key)
                suggested_output_dir_ref["value"] = (
                    None if selected_option is None else selected_option.suggested_export_output_dir
                )
                suggested_output_dir_var.set(
                    "Suggested default: unavailable"
                    if suggested_output_dir_ref["value"] is None
                    else "Suggested default: {}".format(str(suggested_output_dir_ref["value"]))
                )
                if selected_option is not None and str(output_dir_var.get()).strip() == "":
                    if selected_option.suggested_export_output_dir is not None:
                        output_dir_var.set(str(selected_option.suggested_export_output_dir))

            if choice_group.selection_mode == "single" and len(choice_group.options) > 0:
                option_state["selected_option_label_var"].trace_add(
                    "write",
                    _update_single_selection_export_suggestion,
                )

            export_enabled_var.trace_add("write", lambda *_args: _set_export_widgets_state())
            _set_export_widgets_state()

            def _build_selected_option_keys(choice_group=choice_group, option_state=option_state):
                if len(choice_group.options) == 0:
                    return ()
                if choice_group.selection_mode == "multi":
                    return tuple(
                        choice_option.option_key
                        for choice_option in choice_group.options
                        if bool(option_state[choice_option.option_key].get())
                    )
                selected_option_label = option_state["selected_option_label_var"].get()
                selected_option_key = option_state["option_key_by_label"].get(selected_option_label)
                if selected_option_key is None:
                    return ()
                return (selected_option_key,)

            def _parse_export_settings():
                if not bool(export_enabled_var.get()):
                    return None
                output_dir_text = str(output_dir_var.get()).strip()
                if output_dir_text == "":
                    raise ValueError("Plotly export output directory cannot be empty.")
                file_formats = normalize_plotly_export_file_formats(
                    tuple(fragment.strip() for fragment in file_formats_var.get().split(","))
                )
                width = int(width_var.get())
                height = int(height_var.get())
                scale = float(scale_var.get())
                if width <= 0 or height <= 0 or scale <= 0:
                    raise ValueError("Plotly export width, height, and scale must be positive.")
                return RenderBrokerExportSettings(
                    output_dir=Path(output_dir_text),
                    file_formats=file_formats,
                    width=width,
                    height=height,
                    scale=scale,
                )

            def _submit_group_selection(choice_group=choice_group):
                selected_option_keys = _build_selected_option_keys(choice_group=choice_group)
                if len(selected_option_keys) == 0:
                    messagebox.showwarning(
                        "No selection",
                        "Select at least one render option before continuing.",
                        parent=root,
                    )
                    return

                render_backend = resolve_render_backend(
                    bool(backend_vars["open3d"].get()),
                    bool(backend_vars["plotly"].get()),
                )
                try:
                    export_settings = _parse_export_settings()
                except Exception as exc:
                    messagebox.showwarning(
                        "Invalid export settings",
                        str(exc),
                        parent=root,
                    )
                    return

                if render_backend == "none" and export_settings is None:
                    messagebox.showwarning(
                        "No render target selected",
                        "Enable Open3D, Plotly figures, or Plotly export before rendering.",
                        parent=root,
                    )
                    return

                dialog_result["result"] = RenderBrokerDialogResult(
                    decision=RenderBrokerDecision(
                        action="render",
                        group_key=choice_group.group_key,
                        selected_option_keys=selected_option_keys,
                        render_backend=render_backend,
                        export_settings=export_settings,
                    ),
                    session_state=dialog_session_state_ref["value"],
                )
                root.destroy()

            ttk.Button(
                group_frame,
                text=choice_group.render_action_label,
                command=_submit_group_selection,
            ).grid(row=option_controls_row, column=0, sticky="w")

        footer_frame = ttk.Frame(scrollable_frame)
        footer_frame.grid(row=current_row_index, column=0, sticky="ew", pady=(6, 0))
        footer_frame.columnconfigure(0, weight=1)

        timeout_status_var = tk.StringVar(value="")
        timeout_deadline_ref = {"value": None}
        timeout_active_ref = {"value": False}

        def _submit_continue_action() -> None:
            dialog_result["result"] = RenderBrokerDialogResult(
                decision=RenderBrokerDecision(action="continue"),
                session_state=dialog_session_state_ref["value"],
            )
            root.destroy()

        if resolved_timeout_policy is not None and resolved_timeout_policy.timeout_seconds is not None:
            timeout_active_ref["value"] = True
            timeout_deadline_ref["value"] = time.monotonic() + float(resolved_timeout_policy.timeout_seconds)

            ttk.Label(
                footer_frame,
                textvariable=timeout_status_var,
                justify="left",
            ).grid(row=0, column=0, sticky="w", padx=(0, 10))

            def _refresh_timeout_status() -> None:
                if not timeout_active_ref["value"]:
                    return
                remaining_seconds = timeout_deadline_ref["value"] - time.monotonic()
                if remaining_seconds <= 0:
                    timeout_status_var.set("Render dialog timed out; continuing execution.")
                    dialog_result["result"] = RenderBrokerDialogResult(
                        decision=RenderBrokerDecision(action=str(resolved_timeout_policy.timeout_action)),
                        session_state=dialog_session_state_ref["value"],
                    )
                    root.destroy()
                    return
                timeout_status_var.set(
                    "Auto-continue in {:.0f}s if no action is taken.".format(remaining_seconds)
                )
                root.after(250, _refresh_timeout_status)

            if resolved_timeout_policy.allow_extend_timeout:
                def _extend_timeout() -> None:
                    timeout_deadline_ref["value"] = (
                        time.monotonic() + float(resolved_timeout_policy.extend_timeout_seconds)
                    )

                ttk.Button(
                    footer_frame,
                    text="More time",
                    command=_extend_timeout,
                ).grid(row=0, column=1, sticky="e", padx=(0, 8))

            if resolved_timeout_policy.allow_disable_timeout_for_run:
                def _disable_timeout_for_run() -> None:
                    timeout_active_ref["value"] = False
                    dialog_session_state_ref["value"] = replace(
                        dialog_session_state_ref["value"],
                        timeout_disabled_for_run=True,
                    )
                    timeout_status_var.set("Timeout disabled for the rest of this run.")

                ttk.Button(
                    footer_frame,
                    text="Wait indefinitely this run",
                    command=_disable_timeout_for_run,
                ).grid(row=0, column=2, sticky="e", padx=(0, 8))

            _refresh_timeout_status()

        ttk.Button(
            footer_frame,
            text=request.continue_button_label,
            command=_submit_continue_action,
        ).grid(row=0, column=3, sticky="e")

        root.protocol("WM_DELETE_WINDOW", _submit_continue_action)
        try:
            root.mainloop()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

        return dialog_result["result"]