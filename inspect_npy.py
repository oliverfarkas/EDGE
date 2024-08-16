import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import glob
import re
import librosa

class NpyViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Select Directory and Plot .npy and .wav Files")

        # Initialize file list, index, and current row index
        self.files = []
        self.wav_files = {}
        self.current_index = 0
        self.current_row = 0
        self.last_directory = os.path.expanduser("~")  # Default to user's home directory

        # Variables to store plot state
        self.initial_xlim = None
        self.initial_ylim = None
        self.current_xlim = None
        self.current_ylim = None

        # Create GUI components
        self.button_open = tk.Button(root, text="Open Directory", command=self.open_directory)
        self.button_open.pack(pady=10)

        self.frame_plot = tk.Frame(root)
        self.frame_plot.pack(fill='both', expand=True)

        self.button_prev_file = tk.Button(root, text="Previous Slice", command=self.prev_file, state=tk.DISABLED)
        self.button_prev_file.pack(side=tk.LEFT, padx=5)

        self.button_next_file = tk.Button(root, text="Next Slice", command=self.next_file, state=tk.DISABLED)
        self.button_next_file.pack(side=tk.RIGHT, padx=5)

        self.button_prev_line = tk.Button(root, text="Previous Line", command=self.prev_line, state=tk.DISABLED)
        self.button_prev_line.pack(side=tk.LEFT, padx=5)

        self.button_next_line = tk.Button(root, text="Next Line", command=self.next_line, state=tk.DISABLED)
        self.button_next_line.pack(side=tk.RIGHT, padx=5)

        # Checkboxes for displaying data
        self.var_display_npy = tk.BooleanVar(value=True)
        self.var_display_waveform = tk.BooleanVar(value=True)
        self.check_npy = tk.Checkbutton(root, text="Display .npy Data", variable=self.var_display_npy, command=self.update_plot)
        self.check_npy.pack(pady=5)
        self.check_waveform = tk.Checkbutton(root, text="Display Waveform", variable=self.var_display_waveform, command=self.update_plot)
        self.check_waveform.pack(pady=5)

        # Labels for displaying plot information
        self.label_info = tk.Label(root, text="", font=("Arial", 12))
        self.label_info.pack(pady=10)

        self.label_mouse_coords = tk.Label(root, text="Mouse Coordinates: (0, 0)", font=("Arial", 12))
        self.label_mouse_coords.pack(pady=10)

        # Initialize the lines for mouse interaction
        self.hline = None
        self.vline = None
        self.cursor = None

        # Bind arrow keys for navigation
        self.root.bind('<Left>', lambda event: self.prev_file())
        self.root.bind('<Right>', lambda event: self.next_file())

    def open_directory(self):
        directory = filedialog.askdirectory(title="Select Directory", initialdir=self.last_directory)
        if directory:
            self.last_directory = directory
            # Load .npy files and corresponding .wav files
            self.files = self.sort_files_numerically(glob.glob(os.path.join(directory, '*_slice*.npy')))
            self.wav_files = self.sort_files_numerically(glob.glob(os.path.join(directory, '*_slice*.wav')))
            self.current_index = 0
            self.current_row = 0
            if self.files:
                self.update_buttons()
                self.plot_file(self.files[self.current_index])

    def sort_files_numerically(self, file_list):
        # Extract slice numbers and sort files based on these numbers
        def extract_slice_number(filename):
            match = re.search(r'_slice(\d+)\.', filename)
            return int(match.group(1)) if match else float('inf')
        
        return sorted(file_list, key=extract_slice_number)

    def plot_file(self, file_path):
        # Load the .npy file
        data = np.load(file_path)

        # Inspect the contents
        print("Data:", data)
        print("Shape:", data.shape)
        print("Data type:", data.dtype)

        num_rows = data.shape[0]
        colors = cm.viridis(np.linspace(0, 1, num_rows))  # You can choose different colormaps

        # Extract filename from the file path
        filename = os.path.basename(file_path)

        # Ensure current_row is within the new file's range
        if self.current_row >= num_rows:
            self.current_row = num_rows - 1

        # Clear the previous plot
        for widget in self.frame_plot.winfo_children():
            widget.destroy()

        # Create a new figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 6))

        # Initialize the lines for mouse interaction
        self.hline = self.ax.axhline(y=np.nan, color='r', linestyle='--')  # Horizontal line
        self.vline = self.ax.axvline(x=np.nan, color='b', linestyle='--')  # Vertical line

        min_value = np.min(data[self.current_row, :])
        max_value = np.max(data[self.current_row, :])

        if self.var_display_waveform.get():
            # Plot the waveform of the corresponding .wav file
            wav_file_name = filename.replace('.npy', '.wav')
            wav_file_path = os.path.join(os.path.dirname(file_path), wav_file_name)
            try:
                # Load the waveform data
                waveform, samplerate = librosa.load(wav_file_path, sr=960)
                time = np.linspace(0, len(waveform), num=len(waveform))

                # Scale waveform to match the min_value and max_value of .npy data
                waveform = waveform * (max_value - min_value)

                # Plot the waveform
                self.ax.plot(time, waveform, color='red', label='Waveform')
            except Exception as e:
                print(f"Error loading waveform: {e}")
                self.ax.set_xlim([0, data.shape[1]])
        else:
            # Set x-axis limits if no waveform data is plotted
            self.ax.set_xlim([0, data.shape[1]])

        if self.var_display_npy.get():
            # Plot the .npy data
            self.ax.plot(data[self.current_row, :], color=colors[self.current_row], alpha=0.7, label='Numpy Data')  # Adjust alpha for better visualization


        # Add legend and labels
        self.ax.legend()
        self.ax.set_title(f'{filename} - Line {self.current_row}')
        self.ax.set_xlabel('Column Index / Time (s)')
        self.ax.set_ylabel('Value')

        # Store initial plot limits
        if self.initial_xlim is None and self.initial_ylim is None:
            self.initial_xlim = self.ax.get_xlim()
            self.initial_ylim = self.ax.get_ylim()

        # Restore current plot limits if available
        if self.current_xlim is not None and self.current_ylim is not None:
            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)

        # Embed the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add the Matplotlib Navigation Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.frame_plot)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Connect the event handler for mouse movement
        self.cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Update label with plot information
        self.label_info.config(text=f'Slice: {self.current_index}\nLine: {self.current_row}\nMin Value: {min_value:.2f}\nMax Value: {max_value:.2f}')

    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.label_mouse_coords.config(text=f"Mouse Coordinates: ({x:.2f}, {y:.2f})")

            # Update the horizontal and vertical lines
            if self.hline:
                self.hline.set_ydata(y)
            if self.vline:
                self.vline.set_xdata(x)
            
            self.fig.canvas.draw_idle()

    def update_plot(self):
        if self.files:
            self.plot_file(self.files[self.current_index])

    def update_buttons(self):
        if self.files:
            num_rows = np.load(self.files[self.current_index]).shape[0]
            self.button_prev_file.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
            self.button_next_file.config(state=tk.NORMAL if self.current_index < len(self.files) - 1 else tk.DISABLED)
            self.button_prev_line.config(state=tk.NORMAL if self.current_row > 0 else tk.DISABLED)
            self.button_next_line.config(state=tk.NORMAL if self.current_row < num_rows - 1 else tk.DISABLED)
        else:
            self.button_prev_file.config(state=tk.DISABLED)
            self.button_next_file.config(state=tk.DISABLED)
            self.button_prev_line.config(state=tk.DISABLED)
            self.button_next_line.config(state=tk.DISABLED)

    def prev_file(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.current_row = 0  # Reset to first row when changing slices
            self.update_plot()
            self.update_buttons()

    def next_file(self):
        if self.current_index < len(self.files) - 1:
            self.current_index += 1
            num_rows = np.load(self.files[self.current_index]).shape[0]
            self.current_row = min(self.current_row, num_rows - 1)
            self.update_plot()
            self.update_buttons()

    def prev_line(self):
        num_rows = np.load(self.files[self.current_index]).shape[0]
        if self.current_row > 0:
            self.current_row -= 1
            self.update_plot()
            self.update_buttons()

    def next_line(self):
        num_rows = np.load(self.files[self.current_index]).shape[0]
        if self.current_row < num_rows - 1:
            self.current_row += 1
            self.update_plot()
            self.update_buttons()

# Set up the main application window
root = tk.Tk()
app = NpyViewer(root)

# Run the Tkinter event loop
root.mainloop()
