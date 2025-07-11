#!/usr/bin/env python3
"""
eeg_fractal_simulator.py

A Fractal World Generator based on EEG data in a Tkinter GUI:
 - Pane: live fractal world (trees, ferns, stones, clouds, rivers, mountains) modulated by EEG frequencies
 - Controls for loading EEG, playback, and generation
 - Lowest waves (delta) for furthest objects (mountains), higher for closer (trees, ferns)
 - Each moment (EEG window) creates a new image during playback
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mne
from scipy.signal import welch
import time
from collections import deque

# === Procedural fractal world functions ===
def midpoint_displacement(n=512, roughness=1.0, eeg_heights=None):
    arr = np.zeros(n+1)
    arr[0], arr[-1] = random.uniform(0.2,0.8), random.uniform(0.2,0.8)
    step, scale = n, roughness
    while step > 1:
        half = step//2
        for i in range(0,n,step):
            if eeg_heights is not None:
                # Use EEG to influence displacement
                idx = i + half
                if idx < len(eeg_heights):
                    displ = eeg_heights[idx] * scale
                else:
                    displ = random.uniform(-scale, scale)
            else:
                displ = random.uniform(-scale, scale)
            mid = (arr[i] + arr[i+step])/2 + displ
            arr[i+half] = mid
        scale *= 0.6
        step //= 2
    return arr

fern_rules = [
    (0.85, 0.04, -0.04, 0.85, 0, 1.6),
    (0.20, -0.26, 0.23, 0.22, 0, 1.6),
    (-0.15, 0.28, 0.26, 0.24, 0, 0.44),
    (0.00, 0.00, 0.00, 0.16, 0, 0)
]

def draw_fern(ax, x0, y0, scale=50, n=1000):
    x, y = 0, 0
    pts = []
    for _ in range(n):
        a, b, c, d, e, f = random.choice(fern_rules)
        x, y = a*x + b*y + e, c*x + d*y + f
        pts.append((x0 + x*scale, y0 + y*scale))
    arr = np.array(pts)
    ax.scatter(arr[:,0], arr[:,1], s=0.1, c='#228B22', alpha=0.8)

def draw_tree(ax, x, y, angle, depth, length, color_trunk='saddlebrown'):
    if depth == 0:
        return
    rad = np.deg2rad(angle)
    x2 = x + np.cos(rad) * length
    y2 = y + np.sin(rad) * length
    ax.plot([x, x2], [y, y2], c=color_trunk, lw=depth)
    draw_tree(ax, x2, y2, angle + random.uniform(15, 30), depth - 1, length * 0.7)
    draw_tree(ax, x2, y2, angle - random.uniform(15, 30), depth - 1, length * 0.7)

def draw_river(ax, x0, x1, y0, y1, roughness=0.02, gen=5):
    pts = [(x0, y0), (x1, y1)]
    def subdiv(points, g):
        if g == 0:
            return points
        new = []
        for p0, p1 in zip(points, points[1:]):
            x, y = p0
            x2, y2 = p1
            mx, my = (x + x2)/2, (y + y2)/2
            dx, dy = x2 - x, y2 - y
            perp = np.array([-dy, dx])
            perp /= np.linalg.norm(perp) if np.linalg.norm(perp) > 0 else 1
            d = random.uniform(-roughness, roughness)
            new.append((x, y))
            new.append((mx + perp[0]*d, my + perp[1]*d))
        new.append(points[-1])
        return subdiv(new, g-1)
    path = subdiv(pts, gen)
    xs, ys = zip(*path)
    ax.plot(xs, ys, c='steelblue', lw=2, alpha=0.6)

# === EEG Fractal Simulator ===
class EEGFractalSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Fractal Simulator")
        self.raw = None
        self.sfreq = 0
        self.duration = 0
        self.channels = []
        self.current_time = 0.0
        self.is_playing = False
        self.last_update = time.time()
        self.playback_speed = 1.0
        self.window_size = 2.0  # seconds
        self.update_interval = 50  # ms
        self.channel_var = tk.StringVar()
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        self._build_gui()

    def _build_gui(self):
        # Controls
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(fill=tk.X, pady=10)

        ttk.Button(ctrl_frame, text="Load EEG", command=self._load_eeg).pack(side=tk.LEFT, padx=5)
        self.file_info = ttk.Label(ctrl_frame, text="No EEG loaded")
        self.file_info.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Channel:").pack(side=tk.LEFT, padx=5)
        self.channel_combo = ttk.Combobox(ctrl_frame, textvariable=self.channel_var, width=15)
        self.channel_combo.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(ctrl_frame, text="Play", command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Time (s):").pack(side=tk.LEFT, padx=5)
        self.time_var = tk.DoubleVar()
        self.time_slider = ttk.Scale(ctrl_frame, from_=0, to=100, variable=self.time_var, orient=tk.HORIZONTAL, length=200)
        self.time_slider.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(ctrl_frame, from_=0.1, to=5.0, variable=self.speed_var, orient=tk.HORIZONTAL, length=100, command=self._update_speed).pack(side=tk.LEFT, padx=5)

        # Fractal canvas
        self.fig = plt.Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _load_eeg(self):
        filepath = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf"), ("All files", "*.*")])
        if not filepath:
            return
        try:
            self.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.sfreq = self.raw.info['sfreq']
            self.duration = self.raw.n_times / self.sfreq
            self.channels = self.raw.ch_names
            self.channel_combo['values'] = self.channels
            if self.channels:
                self.channel_var.set(self.channels[0])
            self.time_slider.configure(to=self.duration)
            self.file_info.config(text=f"Loaded: {len(self.channels)} ch, {self.duration:.1f}s")
            messagebox.showinfo("Success", "EEG loaded")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _toggle_play(self):
        if not self.raw:
            messagebox.showwarning("Warning", "Load EEG first")
            return
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.last_update = time.time()
            self._update_fractal()

    def _reset(self):
        self.current_time = 0.0
        self.time_var.set(0)
        self.is_playing = False
        self.play_btn.config(text="Play")
        self.ax.cla()
        self.ax.axis('off')
        self.canvas.draw()

    def _update_speed(self, value):
        self.playback_speed = float(value)

    def _get_eeg_window(self):
        if not self.raw or not self.channel_var.get():
            return None
        ch_idx = self.channels.index(self.channel_var.get())
        start_sample = int(self.current_time * self.sfreq)
        window_samples = int(self.window_size * self.sfreq)
        end_sample = min(start_sample + window_samples, self.raw.n_times)
        data, _ = self.raw[ch_idx, start_sample:end_sample]
        return data.flatten()

    def _compute_band_powers(self, data):
        if len(data) < 2:
            return {band: 0.0 for band in self.freq_bands}
        freqs, psd = welch(data, fs=self.sfreq, nperseg=min(256, len(data)))
        powers = {}
        for band, (low, high) in self.freq_bands.items():
            mask = (freqs >= low) & (freqs < high)
            powers[band] = np.sum(psd[mask]) if np.any(mask) else 0.0
        total = sum(powers.values()) + 1e-8
        for band in powers:
            powers[band] /= total  # Normalize
        return powers

    def _compute_eeg_heights(self, data, band='delta', n=512):
        # Filter to band and resample to n points for heights
        low, high = self.freq_bands[band]

        filtered = mne.filter.filter_data(data, self.sfreq, low, high)
        if len(filtered) < 2:
            return np.random.uniform(-0.5, 0.5, n+1)
        # Resample to n+1 points, normalize to [-0.5, 0.5]
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(filtered))
        x_new = np.linspace(0, 1, n+1)
        interp = interp1d(x_old, filtered, kind='linear')
        heights = interp(x_new)
        heights = (heights - np.min(heights)) / (np.max(heights) - np.min(heights) + 1e-8) - 0.5
        return heights

    def _generate_fractal(self):
        data = self._get_eeg_window()
        if data is None or len(data) < 2:
            return
        powers = self._compute_band_powers(data)
        self.ax.cla()
        self.ax.axis('off')

        # Sky (modulated by alpha for calmness)
        sky_intensity = 0.5 + powers['alpha'] * 0.5
        sky = np.linspace(sky_intensity, 1, 200)
        self.ax.imshow(np.vstack([sky, sky]), extent=[0, 1, 0.4, 1], aspect='auto', cmap='Blues')

        # Clouds (number based on gamma power for "noise")
        num_clouds = int(3 + powers['gamma'] * 10)
        for _ in range(num_clouds):
            cx, cy = random.uniform(0.1, 0.9), random.uniform(0.7, 0.95)
            for _ in range(random.randint(3, 8)):
                dx, dy = random.uniform(-0.05, 0.05), random.uniform(-0.02, 0.02)
                r = random.uniform(0.03, 0.07)
                self.ax.add_patch(mpatches.Ellipse((cx + dx, cy + dy), r, r * 0.6, color='white', alpha=0.5))

        # Mountains (lowest waves: delta, furthest)
        roughness_mtn = 0.3 + powers['delta'] * 0.5
        eeg_heights = self._compute_eeg_heights(data, 'delta')
        m = midpoint_displacement(512, roughness_mtn, eeg_heights)
        xs = np.linspace(0, 1, len(m))
        self.ax.fill_between(xs, m * 0.1 + 0.4, 0.4, color='grey')

        # Rivers (theta for flow/roughness)
        num_rivers = int(1 + powers['theta'] * 3)
        roughness_river = 0.02 + powers['theta'] * 0.05
        for _ in range(num_rivers):
            px = random.uniform(0.1, 0.9)
            idx = int(px * (len(m) - 1))
            y0 = m[idx] * 0.1 + 0.4
            draw_river(self.ax, px, px + random.uniform(-0.1, 0.1), y0, 0.0, roughness_river)

        # Ground (base modulated by alpha)
        ground_level = 0.4 - powers['alpha'] * 0.05
        self.ax.fill_between([0, 1], [0, 0], [ground_level, ground_level], color='darkgreen')

        # Ferns (high freq gamma for details, closest)
        num_ferns = int(20 + powers['gamma'] * 50)
        for _ in range(num_ferns):
            draw_fern(self.ax, random.uniform(0, 1), random.uniform(0.02, ground_level - 0.1), scale=random.uniform(0.02, 0.05), n=500)

        # Stones (beta for mid-level details)
        num_stones = int(100 + powers['beta'] * 200)
        for _ in range(num_stones):
            x, y = random.uniform(0, 1), random.uniform(0, ground_level - 0.1)
            size = random.uniform(0.005, 0.02)
            pts = midpoint_displacement(8, 0.2)
            ang = np.linspace(0, 2 * np.pi, len(pts), endpoint=False)
            xs = x + size * np.cos(ang) * pts
            ys = y + size * np.sin(ang) * pts
            self.ax.add_patch(mpatches.Polygon(np.column_stack((xs, ys)), closed=True, color='dimgray'))

        # Trees (beta for structure, closer)
        num_trees = int(20 + powers['beta'] * 50)
        for _ in range(num_trees):
            draw_tree(self.ax, random.uniform(0, 1), ground_level, 90, random.randint(3, 6), random.uniform(0.05, 0.15))

        self.canvas.draw()

    def _update_fractal(self):
        if not self.is_playing:
            return
        self._generate_fractal()
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        self.current_time += dt * self.playback_speed
        if self.current_time >= self.duration:
            self.current_time = 0
        self.time_var.set(self.current_time)
        self.root.after(self.update_interval, self._update_fractal)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = EEGFractalSimulator(root)
    root.mainloop()