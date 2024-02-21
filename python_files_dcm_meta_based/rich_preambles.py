from datetime import datetime, timedelta
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.panel import Panel
from rich.console import Group
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
import time


def get_completed_progress():
    completed_progress = Progress(
                TextColumn(':heavy_check_mark:'),
                TextColumn("{task.description}"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
    return completed_progress

def get_patients_progress(spinner_type):
    patients_progress = Progress(
        SpinnerColumn(spinner_type),
        #*Progress.get_default_columns(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        #TimeRemainingColumn(),
        TextColumn("[green]Patient:"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    return patients_progress

def get_structures_progress(spinner_type):
    structures_progress = Progress(
        SpinnerColumn(spinner_type),
        #*Progress.get_default_columns(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        #TimeRemainingColumn(),
        #TextColumn("[green]Structure:"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    return structures_progress

def get_completed_biopsies_progress():
    completed_biopsies_progress = Progress(
        TextColumn(':heavy_check_mark:'),
        #*Progress.get_default_columns(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        #TimeRemainingColumn(),
        #TextColumn("[green]Biopsy:"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    return completed_biopsies_progress

def get_biopsies_progress(spinner_type):
    biopsies_progress = Progress(
        SpinnerColumn(spinner_type),
        #*Progress.get_default_columns(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        #TimeRemainingColumn(),
        #TextColumn("[green]Biopsy:"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    return biopsies_progress

def get_indeterminate_progress_main(spinner_type):
    indeterminate_progress_main = Progress(
        SpinnerColumn(spinner_type),
        #*Progress.get_default_columns(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        #TaskProgressColumn(),
        #TimeRemainingColumn(),
        TimeElapsedColumn(),
    )
    return indeterminate_progress_main

def get_completed_indeterminate_progress_main():
    completed_indeterminate_progress_main = Progress(
        TextColumn(':heavy_check_mark:'),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    )
    return completed_indeterminate_progress_main

def get_indeterminate_progress_sub(spinner_type):
    indeterminate_progress_sub = Progress(
        SpinnerColumn(spinner_type),
        #*Progress.get_default_columns(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        #TaskProgressColumn(),
        #TimeRemainingColumn(),
        TimeElapsedColumn(),
    )
    return indeterminate_progress_sub


def get_MC_trial_progress(spinner_type):
    MC_trial_progress = Progress(
        SpinnerColumn(spinner_type),
        #*Progress.get_default_columns(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        #TimeRemainingColumn(),
        #TextColumn("[green]MC trial:"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    return MC_trial_progress


def get_progress_group(completed_progress, 
patients_progress, indeterminate_progress_main, 
structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_sub):
    
    progress_group = Panel(
        Group(
            Panel(Group(completed_progress), title="Completed main tasks", title_align='left'),
            Panel(Group(patients_progress,indeterminate_progress_main), title="In progress main tasks", title_align='left'),
            Panel(Group(biopsies_progress, structures_progress, MC_trial_progress, indeterminate_progress_sub), title="In progress subtasks", title_align='left')
        ), 
        title="Algorithm Progress", title_align='left'
    )
    return progress_group


def get_progress_all(spinner_type):
    completed_progress = get_completed_progress()
    patients_progress = get_patients_progress(spinner_type)
    structures_progress = get_structures_progress(spinner_type)
    completed_biopsies_progress = get_completed_biopsies_progress()
    biopsies_progress = get_biopsies_progress(spinner_type)
    indeterminate_progress_main = get_indeterminate_progress_main(spinner_type)
    completed_indeterminate_progress_main = get_completed_indeterminate_progress_main()
    indeterminate_progress_sub = get_indeterminate_progress_sub(spinner_type)
    MC_trial_progress = get_MC_trial_progress(spinner_type)

    progress_group = get_progress_group(
        completed_progress, patients_progress, indeterminate_progress_main, 
        structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_sub
        )

    return completed_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group



class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            "Biopsy [bold green]LocaliZer[/bold green] Application",
            datetime.now().ctime().replace(":", "[blink]:[/]"),
        )
        return Panel(grid)


class info_output:
    """display important information"""
    def __init__(self):
        self.text_important_Text = Text()
        self.line_num = 1

    def __rich__(self) -> Panel:
        return Panel(self.text_important_Text, title="Important information", title_align='left')

    def add_text_line(self,text_str, live_display_obj):
        self.text_important_Text.append("[{}]".format(self.line_num), style = 'cyan')
        self.text_important_Text.append("[{}]".format(datetime.now().strftime("%H:%M:%S")), style = 'magenta')
        self.text_important_Text.append("> "+text_str+"\n")
        self.line_num = self.line_num + 1
        live_display_obj.refresh() # refresh the live display everytime you add a text line



class Footer:
    """Display footer with elapsed and calculation time."""
    def __init__(self,algo_global_start_time, stopwatch):
        self.algo_global_start_time = algo_global_start_time
        self.stopwatch = stopwatch
    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        elapsed_seconds = time.time()-self.algo_global_start_time
        elapsed_seconds_rounded = round(elapsed_seconds)
        elapsed_delta_time = timedelta(seconds=elapsed_seconds_rounded)

        calculation_seconds = self.stopwatch.duration
        calculation_seconds_rounded = round(calculation_seconds)
        calculation_delta_time = timedelta(seconds=calculation_seconds_rounded)

        grid.add_row(
            "[bold magenta]Total elapsed time (H:MM:SS): {}".format(elapsed_delta_time)+",    "+"[bold magenta]Total calculation time (H:MM:SS): {}".format(calculation_delta_time),
            "Developed by: MJM"
        )
        return Panel(grid)

def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", minimum_size=3, size=3),
        Layout(name="main"),
        Layout(name="footer", minimum_size=3, size=3),
    )
    layout["main"].split_row(
        Layout(name="main-left"),
        Layout(name="main-right"),
    )
    #layout["side"].split(Layout(name="box1"), Layout(name="box2"))
    return layout


