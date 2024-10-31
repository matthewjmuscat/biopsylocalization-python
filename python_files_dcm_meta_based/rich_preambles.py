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
import psutil
import GPUtil


def get_completed_progress():
    completed_progress = Progress(
                TextColumn(':heavy_check_mark:'),
                TextColumn("{task.description}"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
    return completed_progress

def get_completed_sections_progress():
    completed_progress = Progress(
                TextColumn(':heavy_check_mark:'),
                TextColumn("{task.description}")
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

"""
def get_progress_group(completed_progress, 
patients_progress, indeterminate_progress_main, 
structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_sub):
    
    progress_group = Panel(
        Group(
            Panel(Group(completed_progress), title="Completed section tasks", title_align='left'),
            Panel(Group(patients_progress,indeterminate_progress_main), title="In progress section tasks", title_align='left'),
            Panel(Group(biopsies_progress, structures_progress, MC_trial_progress, indeterminate_progress_sub), title="In progress subtasks", title_align='left')
        ), 
        title="Algorithm Progress", title_align='left'
    )
    return progress_group
"""

def get_progress_group(completed_progress, 
                       completed_sections_progress,  # Add new panel here
                       patients_progress, 
                       indeterminate_progress_main, 
                       structures_progress, 
                       biopsies_progress, 
                       MC_trial_progress, 
                       indeterminate_progress_sub):
    
    progress_group = Panel(
        Group(
            Panel(Group(completed_sections_progress), title="Completed sections", title_align='left'),  # New section
            Panel(Group(completed_progress), title="Completed section tasks", title_align='left'),
            Panel(Group(patients_progress,indeterminate_progress_main), title="In progress section tasks", title_align='left'),
            Panel(Group(biopsies_progress, structures_progress, MC_trial_progress, indeterminate_progress_sub), title="In progress subtasks", title_align='left')
        ), 
        title="Algorithm Progress", title_align='left'
    )
    return progress_group


"""
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
        completed_progress, completed_sections_progress, patients_progress, indeterminate_progress_main, 
        structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_sub
        )

    return completed_progress, completed_sections_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group
"""

def get_progress_all(spinner_type):
    completed_progress = get_completed_progress()
    completed_sections_progress = get_completed_sections_progress()  # For completed sections
    patients_progress = get_patients_progress(spinner_type)
    structures_progress = get_structures_progress(spinner_type)
    completed_biopsies_progress = get_completed_biopsies_progress()
    biopsies_progress = get_biopsies_progress(spinner_type)
    indeterminate_progress_main = get_indeterminate_progress_main(spinner_type)
    completed_indeterminate_progress_main = get_completed_indeterminate_progress_main()
    indeterminate_progress_sub = get_indeterminate_progress_sub(spinner_type)
    MC_trial_progress = get_MC_trial_progress(spinner_type)

    # Update the group to include the new completed sections panel
    progress_group = get_progress_group(
        completed_progress, completed_sections_progress, patients_progress, 
        indeterminate_progress_main, structures_progress, 
        biopsies_progress, MC_trial_progress, indeterminate_progress_sub
    )

    return (completed_progress, completed_sections_progress,  # Return completed sections too
            patients_progress, structures_progress, biopsies_progress, 
            MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group)


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
    def __init__(self, max_lines=20):
        self.text_lines = []  # Store lines as a list of strings
        #self.text_important_Text = Text()
        self.line_num = 1
        self.max_lines = max_lines

    """
    def __rich__(self) -> Panel:
        return Panel(self.text_important_Text, title="Important information", title_align='left')
    """

    def __rich__(self) -> Panel:
        combined_text = Text()  # Combine all Text objects into one for display
        for text_obj in self.text_lines:
            combined_text.append(text_obj)
        return Panel(combined_text, title="Important information", title_align='left')

    """
    def add_text_line(self,text_str, live_display_obj):
        self.text_important_Text.append("[{}]".format(self.line_num), style = 'cyan')
        self.text_important_Text.append("[{}]".format(datetime.now().strftime("%H:%M:%S")), style = 'magenta')
        self.text_important_Text.append("> "+text_str+"\n")
        self.line_num = self.line_num + 1
        live_display_obj.refresh() # refresh the live display everytime you add a text line
    """
    """
    def add_text_line(self, text_str, live_display_obj):
        if len(self.text_important_Text.lines) >= self.max_lines:
            # Remove the oldest line (first three segments: line number, timestamp, text)
            self.text_important_Text.plain = "\n".join(self.text_important_Text.plain.split("\n")[3:])
        self.text_important_Text.append("[{}] ".format(self.line_num), style='cyan')
        self.text_important_Text.append("[{}] ".format(datetime.now().strftime("%H:%M:%S")), style='magenta')
        self.text_important_Text.append("> "+text_str+"\n")
        self.line_num += 1
        live_display_obj.refresh()  # Refresh the live display every time you add a text line
    """
    """
    def add_text_line(self, text_str, live_display_obj):
        # Compose the new line
        new_line = f"[{self.line_num}] [cyan][{datetime.now().strftime('%H:%M:%S')}][/] > {text_str}\n"
        
        # Add new line to the list
        self.text_lines.append(new_line)

        # Check if the total number of lines exceeds the maximum allowed
        if len(self.text_lines) > self.max_lines:
            # Remove the oldest line
            self.text_lines.pop(0)

        # Update the Text object with the current list of lines
        self.text_important_Text.plain = "".join(self.text_lines)

        # Increment the line number
        self.line_num += 1

        # Refresh the live display
        live_display_obj.refresh()
    """

    def add_text_line(self, text_str, live_display_obj):
        # Create a new Text object with styling for the new line
        new_line = Text()
        new_line.append(f"[{self.line_num}]", style='cyan')
        new_line.append(f"[{datetime.now().strftime('%H:%M:%S')}]", style='magenta')
        new_line.append(f"> {text_str}\n")

        # Add the new Text object to the list
        self.text_lines.append(new_line)

        # Check if the total number of lines exceeds the maximum allowed
        if len(self.text_lines) > self.max_lines:
            # Remove the oldest Text object
            self.text_lines.pop(0)

        # Refresh the live display by calling refresh on the live display object
        live_display_obj.refresh()

        # Increment the line number
        self.line_num += 1


"""
class Footer:
    # Display footer with elapsed and calculation time.
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
"""



class Footer:
    """Display footer with elapsed and calculation time, along with CPU, memory, and GPU memory usage."""
    def __init__(self, algo_global_start_time, stopwatch):
        self.algo_global_start_time = algo_global_start_time
        self.stopwatch = stopwatch

    def get_system_usage(self):
        """Get the current CPU, memory, and GPU memory usage."""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent

        # Available memory in GB (bytes to GB conversion)
        free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

        # Get GPU memory info using GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            free_gpu_memory = gpus[0].memoryFree / 1024  # Convert to GB
            total_gpu_memory = gpus[0].memoryTotal / 1024  # Convert to GB
        else:
            free_gpu_memory = None
            total_gpu_memory = None

        return cpu_usage, memory_usage, free_memory_gb, free_gpu_memory, total_gpu_memory

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        
        # Calculate elapsed and calculation time
        elapsed_seconds = time.time() - self.algo_global_start_time
        elapsed_seconds_rounded = round(elapsed_seconds)
        elapsed_delta_time = timedelta(seconds=elapsed_seconds_rounded)

        calculation_seconds = self.stopwatch.duration
        calculation_seconds_rounded = round(calculation_seconds)
        calculation_delta_time = timedelta(seconds=calculation_seconds_rounded)

        # Get system usage including GPU memory
        cpu_usage, memory_usage, free_memory_gb, free_gpu_memory, total_gpu_memory = self.get_system_usage()

        # GPU memory info formatting
        if free_gpu_memory is not None:
            gpu_info = "[bold cyan]GPU Mem: {:.2f}/{:.2f} GB".format(free_gpu_memory, total_gpu_memory)
        else:
            gpu_info = "[bold red]No GPU detected"

        # Add content to the grid
        grid.add_row(
            "[bold magenta]Total elapsed time (H:MM:SS): {}".format(elapsed_delta_time) + ",    " +
            "[bold magenta]Total calculation time (H:MM:SS): {}".format(calculation_delta_time),
            "Developed by: MJM"
        )

        # CPU, RAM, and GPU info all on the same row
        grid.add_row(
            "[bold yellow]CPU usage: {:>4.2f}%".format(cpu_usage) + ",  " +
            "[bold yellow]Memory usage: {:>4.2f}%".format(memory_usage) + ",  " +
            "[bold yellow]Avail. Mem: {:>4.2f} GB".format(free_memory_gb) + ",  " + gpu_info,
            ""
        )

        # Return a panel with the grid
        return Panel(grid)

# Function to define layout
def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", minimum_size=3, size=3),
        Layout(name="main"),
        Layout(name="footer", minimum_size=4, size=4),  
    )
    layout["main"].split_row(
        Layout(name="main-left"),
        Layout(name="main-right"),
    )
    return layout


class CompletedSectionsManager:
    """Manage completed sections and add them to the progress panel."""
    
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar  # Progress bar for completed sections
        self.completed_sections = []  # Store section names and elapsed times

    def add_completed_section(self, section_name, elapsed_time):
        """Add a completed section to the progress bar."""
        self.completed_sections.append((section_name, elapsed_time))
        # Format elapsed_time to remove decimals
        rounded_elapsed_time = timedelta(seconds=round(elapsed_time.total_seconds()))  # Round to nearest second
        formatted_time = str(rounded_elapsed_time)  # Format the timedelta object
        # Add the section to the progress bar
        task_id = self.progress_bar.add_task(f"[green]{section_name} (Elapsed time: {formatted_time})", total=1)
        self.progress_bar.update(task_id, completed=1)



def clear_completed_main_tasks(completed_progress):
    """Clear the completed section tasks section."""
    for task in completed_progress.tasks:
        completed_progress.remove_task(task.id)



def section_completed(section_name, start_time, completed_progress, completed_sections_manager):
    """When a section is completed, clear tasks and add it to completed sections."""
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    # Clear the completed section tasks
    clear_completed_main_tasks(completed_progress)

    # Add the completed section to the completed sections panel
    completed_sections_manager.add_completed_section(section_name, elapsed_time)
