"""
Life-Pulse v2.0 - Disaster Recovery Dual-Mode Radar System
===========================================================
A Raspberry Pi 5 based system for:
  - Mode A (Search): Detecting human heartbeats/breathing (0.2Hz - 2.0Hz)
  - Mode B (Sentry): Detecting structural vibrations/wall shifts (10Hz+)

Author: Embedded Systems Implementation
Hardware: Raspberry Pi 5 with radar sensor, MPU-6050, 5" touchscreen
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import queue

# GUI imports
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class OperationMode(Enum):
    """Operating modes for Life-Pulse system"""
    SEARCH = "SEARCH"   # Human detection mode (0.2Hz - 2.0Hz)
    SENTRY = "SENTRY"   # Structural monitoring mode (10Hz+)


class DetectionStatus(Enum):
    """Detection status states"""
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    SURVIVOR_DETECTED = "SURVIVOR_DETECTED"
    COLLAPSE_WARNING = "COLLAPSE_WARNING"


@dataclass
class SystemConfig:
    """System configuration parameters"""
    # Sampling parameters
    sample_rate: float = 100.0          # Hz
    buffer_size: int = 512              # samples per buffer
    
    # Search mode parameters (heartbeat detection)
    heartbeat_freq_min: float = 0.2     # Hz
    heartbeat_freq_max: float = 2.0     # Hz
    heartbeat_target_freq: float = 1.2  # Hz (typical heartbeat)
    heartbeat_threshold: float = 0.3    # Detection threshold
    
    # Sentry mode parameters (structural monitoring)
    vibration_freq_min: float = 10.0    # Hz
    spike_threshold: float = 2.5        # G-force threshold for alarm
    
    # Signal processing
    noise_amplitude: float = 0.5        # Base noise level
    signal_amplitude: float = 1.0       # Signal amplitude


# =============================================================================
# MOCK SENSOR CLASSES
# =============================================================================

class MockRadarSensor:
    """
    Mock Radar Sensor for testing without hardware.
    
    Generates a noisy sine wave with optional target injection:
    - In SEARCH mode: Injects 1.2Hz sine wave (simulated heartbeat)
    - In SENTRY mode: Injects random high-amplitude spikes (structural shift)
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.mode = OperationMode.SEARCH
        self.target_present = False
        self._time_offset = 0.0
        self._lock = threading.Lock()
        
    def set_mode(self, mode: OperationMode):
        """Set the operating mode"""
        with self._lock:
            self.mode = mode
            
    def set_target_present(self, present: bool):
        """Enable/disable target signal injection"""
        with self._lock:
            self.target_present = present
    
    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        Read simulated radar samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            numpy array of voltage readings (simulated)
        """
        with self._lock:
            dt = 1.0 / self.config.sample_rate
            t = np.arange(num_samples) * dt + self._time_offset
            self._time_offset += num_samples * dt
            
            # Base noise (Gaussian)
            noise = np.random.normal(0, self.config.noise_amplitude, num_samples)
            
            # Base low-frequency drift
            drift = 0.1 * np.sin(2 * np.pi * 0.05 * t)
            
            signal_data = noise + drift
            
            if self.target_present:
                if self.mode == OperationMode.SEARCH:
                    # Inject heartbeat signal (1.2Hz sine wave)
                    heartbeat = self.config.signal_amplitude * np.sin(
                        2 * np.pi * self.config.heartbeat_target_freq * t
                    )
                    # Add slight frequency variation for realism
                    breathing = 0.3 * np.sin(2 * np.pi * 0.25 * t)
                    signal_data += heartbeat + breathing
                    
                elif self.mode == OperationMode.SENTRY:
                    # Inject random high-amplitude spikes (structural shifting)
                    num_spikes = np.random.randint(1, 4)
                    spike_positions = np.random.randint(0, num_samples, num_spikes)
                    for pos in spike_positions:
                        if pos < num_samples - 10:
                            # Create a damped oscillation spike
                            spike_len = min(30, num_samples - pos)
                            decay = np.exp(-np.arange(spike_len) * 0.2)
                            spike_freq = np.random.uniform(15, 30)  # Hz
                            spike = (self.config.spike_threshold + np.random.uniform(0, 1)) * \
                                    np.sin(2 * np.pi * spike_freq * np.arange(spike_len) * dt) * decay
                            signal_data[pos:pos+spike_len] += spike
            
            return signal_data


class MockVibrationSensor:
    """
    Mock MPU-6050 Vibration Sensor.
    
    Simulates accelerometer readings with noise filtering capability.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self._time_offset = 0.0
        self._lock = threading.Lock()
        self.high_vibration = False
        
    def set_high_vibration(self, enabled: bool):
        """Simulate high vibration conditions"""
        with self._lock:
            self.high_vibration = enabled
    
    def read_acceleration(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read simulated 3-axis accelerometer data.
        
        Returns:
            Tuple of (x, y, z) acceleration arrays in G-force
        """
        with self._lock:
            dt = 1.0 / self.config.sample_rate
            t = np.arange(num_samples) * dt + self._time_offset
            self._time_offset += num_samples * dt
            
            # Base sensor noise
            noise_x = np.random.normal(0, 0.02, num_samples)
            noise_y = np.random.normal(0, 0.02, num_samples)
            noise_z = np.random.normal(0, 0.02, num_samples)
            
            # Gravity (Z-axis when flat)
            gravity_z = np.ones(num_samples) * 1.0
            
            if self.high_vibration:
                # Add structural vibration
                vib_freq = np.random.uniform(12, 25)
                vibration = 0.5 * np.sin(2 * np.pi * vib_freq * t)
                noise_z += vibration
                
                # Random spikes
                if np.random.random() > 0.7:
                    spike_pos = np.random.randint(0, num_samples)
                    spike_width = min(20, num_samples - spike_pos)
                    noise_z[spike_pos:spike_pos+spike_width] += np.random.uniform(1, 3)
            
            return (noise_x, noise_y, gravity_z + noise_z)


# =============================================================================
# DIGITAL SIGNAL PROCESSING
# =============================================================================

class SignalProcessor:
    """
    Digital Signal Processing class for Life-Pulse system.
    
    Implements FFT-based frequency analysis and peak detection.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def compute_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of input signal.
        
        Args:
            signal_data: Time-domain signal array
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        n = len(signal_data)
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(n)
        windowed_signal = signal_data * window
        
        # Compute FFT
        fft_result = fft(windowed_signal)
        frequencies = fftfreq(n, 1.0 / self.config.sample_rate)
        
        # Get positive frequencies only
        positive_mask = frequencies >= 0
        frequencies = frequencies[positive_mask]
        magnitudes = np.abs(fft_result[positive_mask]) * 2.0 / n
        
        return frequencies, magnitudes
    
    def bandpass_filter(self, signal_data: np.ndarray, 
                        low_freq: float, high_freq: float) -> np.ndarray:
        """
        Apply bandpass filter to signal.
        
        Args:
            signal_data: Input signal
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Upper cutoff frequency (Hz)
            
        Returns:
            Filtered signal
        """
        nyquist = self.config.sample_rate / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Clamp to valid range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        # Design Butterworth bandpass filter
        order = 4
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter (use filtfilt for zero phase distortion)
        try:
            filtered = signal.filtfilt(b, a, signal_data)
        except ValueError:
            # Fallback if signal is too short
            filtered = signal_data
            
        return filtered
    
    def detect_heartbeat(self, signal_data: np.ndarray) -> Tuple[bool, float, float]:
        """
        Detect heartbeat frequency in signal (Search Mode).
        
        Args:
            signal_data: Raw signal data
            
        Returns:
            Tuple of (detected, peak_frequency, peak_magnitude)
        """
        # Bandpass filter for heartbeat range
        filtered = self.bandpass_filter(
            signal_data,
            self.config.heartbeat_freq_min,
            self.config.heartbeat_freq_max
        )
        
        # Compute FFT
        frequencies, magnitudes = self.compute_fft(filtered)
        
        # Find peaks in the heartbeat frequency range
        freq_mask = (frequencies >= self.config.heartbeat_freq_min) & \
                    (frequencies <= self.config.heartbeat_freq_max)
        
        if not np.any(freq_mask):
            return False, 0.0, 0.0
            
        masked_magnitudes = magnitudes[freq_mask]
        masked_frequencies = frequencies[freq_mask]
        
        # Find peak
        peak_idx = np.argmax(masked_magnitudes)
        peak_freq = masked_frequencies[peak_idx]
        peak_mag = masked_magnitudes[peak_idx]
        
        # Detection threshold
        detected = peak_mag > self.config.heartbeat_threshold
        
        return detected, peak_freq, peak_mag
    
    def detect_structural_shift(self, signal_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect structural vibrations/shifts (Sentry Mode).
        
        Args:
            signal_data: Raw signal data
            
        Returns:
            Tuple of (alarm_triggered, max_amplitude)
        """
        # Highpass filter to remove low-frequency content
        filtered = self.bandpass_filter(
            signal_data,
            self.config.vibration_freq_min,
            self.config.sample_rate / 2.5
        )
        
        # Check for amplitude spikes
        max_amplitude = np.max(np.abs(filtered))
        
        # Also compute RMS for sustained vibration detection
        rms = np.sqrt(np.mean(filtered ** 2))
        
        # Trigger alarm if max amplitude exceeds threshold
        alarm = max_amplitude > self.config.spike_threshold or rms > (self.config.spike_threshold / 2)
        
        return alarm, max_amplitude


# =============================================================================
# DATA ACQUISITION THREAD
# =============================================================================

class DataAcquisitionThread(threading.Thread):
    """
    Background thread for continuous data acquisition and processing.
    """
    
    def __init__(self, config: SystemConfig, 
                 radar_sensor: MockRadarSensor,
                 signal_processor: SignalProcessor,
                 data_queue: queue.Queue,
                 result_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.radar = radar_sensor
        self.processor = signal_processor
        self.data_queue = data_queue
        self.result_queue = result_queue
        self._running = True
        self._lock = threading.Lock()
        
    def stop(self):
        """Stop the acquisition thread"""
        with self._lock:
            self._running = False
            
    def is_running(self) -> bool:
        """Check if thread is running"""
        with self._lock:
            return self._running
    
    def run(self):
        """Main acquisition loop"""
        while self.is_running():
            try:
                # Read sensor data
                raw_signal = self.radar.read_samples(self.config.buffer_size)
                
                # Process based on mode
                mode = self.radar.mode
                
                if mode == OperationMode.SEARCH:
                    detected, freq, mag = self.processor.detect_heartbeat(raw_signal)
                    result = {
                        'mode': mode,
                        'detected': detected,
                        'frequency': freq,
                        'magnitude': mag,
                        'alarm': False
                    }
                else:  # SENTRY mode
                    alarm, amplitude = self.processor.detect_structural_shift(raw_signal)
                    result = {
                        'mode': mode,
                        'detected': False,
                        'frequency': 0.0,
                        'magnitude': amplitude,
                        'alarm': alarm
                    }
                
                # Send raw signal for visualization
                try:
                    self.data_queue.put_nowait(raw_signal)
                except queue.Full:
                    pass
                    
                # Send processing result
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass
                
                # Simulate real-time delay
                time.sleep(self.config.buffer_size / self.config.sample_rate)
                
            except Exception as e:
                print(f"Acquisition error: {e}")
                time.sleep(0.1)


# =============================================================================
# GUI APPLICATION
# =============================================================================

class LifePulseGUI:
    """
    Main GUI Application for Life-Pulse v2.0
    
    Features:
    - Real-time signal visualization
    - Status panel with detection indicators
    - Mode switching (Search/Sentry)
    - Dark theme interface
    """
    
    # Color scheme (dark theme)
    COLORS = {
        'bg_dark': '#1a1a2e',
        'bg_medium': '#16213e',
        'bg_light': '#0f3460',
        'accent': '#e94560',
        'success': '#00ff88',
        'warning': '#ffaa00',
        'danger': '#ff4444',
        'text': '#ffffff',
        'text_dim': '#888888',
        'graph_line': '#00ffff',
        'graph_grid': '#333355'
    }
    
    def __init__(self):
        self.config = SystemConfig()
        
        # Initialize sensors
        self.radar_sensor = MockRadarSensor(self.config)
        self.vibration_sensor = MockVibrationSensor(self.config)
        self.signal_processor = SignalProcessor(self.config)
        
        # Data queues for thread communication
        self.data_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Start acquisition thread
        self.acquisition_thread = DataAcquisitionThread(
            self.config,
            self.radar_sensor,
            self.signal_processor,
            self.data_queue,
            self.result_queue
        )
        
        # GUI state
        self.current_mode = OperationMode.SEARCH
        self.current_status = DetectionStatus.SEARCHING
        self.signal_buffer = np.zeros(self.config.buffer_size * 4)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Life-Pulse v2.0 - Disaster Recovery System")
        self.root.geometry("1024x600")
        self.root.configure(bg=self.COLORS['bg_dark'])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Make it responsive to touch/resize
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self._create_widgets()
        self._start_acquisition()
        
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg_dark'])
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # === Header Panel ===
        header_frame = tk.Frame(main_frame, bg=self.COLORS['bg_medium'], height=80)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🔴 LIFE-PULSE v2.0",
            font=("Helvetica", 24, "bold"),
            fg=self.COLORS['accent'],
            bg=self.COLORS['bg_medium']
        )
        title_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        
        # Subtitle
        subtitle = tk.Label(
            header_frame,
            text="Dual-Mode Disaster Recovery Radar System",
            font=("Helvetica", 10),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_medium']
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")
        
        # Mode indicator
        self.mode_indicator = tk.Label(
            header_frame,
            text="MODE: SEARCH",
            font=("Helvetica", 14, "bold"),
            fg=self.COLORS['success'],
            bg=self.COLORS['bg_light'],
            padx=20,
            pady=10
        )
        self.mode_indicator.grid(row=0, column=2, rowspan=2, padx=20, pady=10)
        
        # === Main Content Area ===
        content_frame = tk.Frame(main_frame, bg=self.COLORS['bg_dark'])
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # === Graph Panel (Left) ===
        graph_frame = tk.Frame(content_frame, bg=self.COLORS['bg_medium'])
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=self.COLORS['bg_medium'])
        self.ax = self.fig.add_subplot(111)
        self._style_plot()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Initialize plot line
        self.line, = self.ax.plot([], [], color=self.COLORS['graph_line'], linewidth=1)
        
        # === Status Panel (Right) ===
        status_frame = tk.Frame(content_frame, bg=self.COLORS['bg_medium'])
        status_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Status title
        status_title = tk.Label(
            status_frame,
            text="SYSTEM STATUS",
            font=("Helvetica", 12, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium']
        )
        status_title.pack(pady=(20, 10))
        
        # Main status display
        self.status_display = tk.Label(
            status_frame,
            text="SEARCHING...",
            font=("Helvetica", 16, "bold"),
            fg=self.COLORS['warning'],
            bg=self.COLORS['bg_dark'],
            wraplength=200,
            justify="center",
            padx=20,
            pady=30
        )
        self.status_display.pack(padx=10, pady=10, fill="x")
        
        # Frequency display
        freq_frame = tk.Frame(status_frame, bg=self.COLORS['bg_medium'])
        freq_frame.pack(padx=10, pady=10, fill="x")
        
        tk.Label(
            freq_frame,
            text="Detected Freq:",
            font=("Helvetica", 10),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_medium']
        ).pack()
        
        self.freq_display = tk.Label(
            freq_frame,
            text="-- Hz",
            font=("Helvetica", 20, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium']
        )
        self.freq_display.pack()
        
        # Magnitude display
        mag_frame = tk.Frame(status_frame, bg=self.COLORS['bg_medium'])
        mag_frame.pack(padx=10, pady=10, fill="x")
        
        tk.Label(
            mag_frame,
            text="Signal Strength:",
            font=("Helvetica", 10),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_medium']
        ).pack()
        
        self.mag_display = tk.Label(
            mag_frame,
            text="--",
            font=("Helvetica", 20, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium']
        )
        self.mag_display.pack()
        
        # Separator
        ttk.Separator(status_frame, orient="horizontal").pack(fill="x", padx=10, pady=20)
        
        # Mode toggle button
        self.mode_button = tk.Button(
            status_frame,
            text="SWITCH TO\nSENTRY MODE",
            font=("Helvetica", 12, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_light'],
            activebackground=self.COLORS['accent'],
            activeforeground=self.COLORS['text'],
            relief="flat",
            command=self.toggle_mode,
            width=15,
            height=3
        )
        self.mode_button.pack(padx=10, pady=10)
        
        # Simulate target toggle (for testing)
        self.target_var = tk.BooleanVar(value=False)
        self.target_check = tk.Checkbutton(
            status_frame,
            text="Simulate Target",
            variable=self.target_var,
            command=self.toggle_target,
            font=("Helvetica", 10),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium'],
            selectcolor=self.COLORS['bg_dark'],
            activebackground=self.COLORS['bg_medium'],
            activeforeground=self.COLORS['text']
        )
        self.target_check.pack(pady=10)
        
        # === Footer ===
        footer_frame = tk.Frame(main_frame, bg=self.COLORS['bg_dark'], height=30)
        footer_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        footer_label = tk.Label(
            footer_frame,
            text="Life-Pulse v2.0 | Raspberry Pi 5 | Mock Sensor Mode",
            font=("Helvetica", 8),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_dark']
        )
        footer_label.pack(side="right")
        
    def _style_plot(self):
        """Apply dark theme styling to the plot"""
        self.ax.set_facecolor(self.COLORS['bg_dark'])
        self.ax.tick_params(colors=self.COLORS['text_dim'])
        self.ax.spines['bottom'].set_color(self.COLORS['graph_grid'])
        self.ax.spines['top'].set_color(self.COLORS['graph_grid'])
        self.ax.spines['left'].set_color(self.COLORS['graph_grid'])
        self.ax.spines['right'].set_color(self.COLORS['graph_grid'])
        self.ax.set_xlabel('Samples', color=self.COLORS['text_dim'])
        self.ax.set_ylabel('Amplitude', color=self.COLORS['text_dim'])
        self.ax.set_title('Live Signal Monitor', color=self.COLORS['text'], fontweight='bold')
        self.ax.grid(True, color=self.COLORS['graph_grid'], alpha=0.3, linestyle='--')
        self.ax.set_xlim(0, self.config.buffer_size * 4)
        self.ax.set_ylim(-5, 5)
        
    def toggle_mode(self):
        """Toggle between Search and Sentry modes"""
        if self.current_mode == OperationMode.SEARCH:
            self.current_mode = OperationMode.SENTRY
            self.mode_indicator.configure(text="MODE: SENTRY", fg=self.COLORS['warning'])
            self.mode_button.configure(text="SWITCH TO\nSEARCH MODE")
            self.status_display.configure(text="MONITORING...", fg=self.COLORS['warning'])
        else:
            self.current_mode = OperationMode.SEARCH
            self.mode_indicator.configure(text="MODE: SEARCH", fg=self.COLORS['success'])
            self.mode_button.configure(text="SWITCH TO\nSENTRY MODE")
            self.status_display.configure(text="SEARCHING...", fg=self.COLORS['warning'])
            
        self.radar_sensor.set_mode(self.current_mode)
        
    def toggle_target(self):
        """Toggle target simulation"""
        self.radar_sensor.set_target_present(self.target_var.get())
        self.vibration_sensor.set_high_vibration(self.target_var.get())
        
    def _start_acquisition(self):
        """Start data acquisition"""
        self.acquisition_thread.start()
        self._update_gui()
        
    def _update_gui(self):
        """Update GUI with latest data (runs in main thread)"""
        try:
            # Get latest signal data
            while not self.data_queue.empty():
                new_data = self.data_queue.get_nowait()
                # Shift buffer and append new data
                self.signal_buffer = np.roll(self.signal_buffer, -len(new_data))
                self.signal_buffer[-len(new_data):] = new_data
                
            # Update plot
            x_data = np.arange(len(self.signal_buffer))
            self.line.set_data(x_data, self.signal_buffer)
            self.ax.set_xlim(0, len(self.signal_buffer))
            
            # Auto-scale Y axis based on data
            data_range = np.max(np.abs(self.signal_buffer))
            if data_range > 0:
                self.ax.set_ylim(-data_range * 1.2, data_range * 1.2)
            
            self.canvas.draw_idle()
            
            # Get latest detection results
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self._update_status(result)
                
        except Exception as e:
            print(f"GUI update error: {e}")
            
        # Schedule next update
        self.root.after(50, self._update_gui)  # 20 FPS update rate
        
    def _update_status(self, result: dict):
        """Update status displays based on detection results"""
        if result['mode'] == OperationMode.SEARCH:
            # Update frequency display
            if result['detected']:
                self.freq_display.configure(text=f"{result['frequency']:.2f} Hz")
                self.mag_display.configure(text=f"{result['magnitude']:.3f}")
                self.status_display.configure(
                    text="🚨 SURVIVOR\nDETECTED!",
                    fg=self.COLORS['success']
                )
                self.current_status = DetectionStatus.SURVIVOR_DETECTED
            else:
                self.freq_display.configure(text="-- Hz")
                self.mag_display.configure(text=f"{result['magnitude']:.3f}")
                self.status_display.configure(
                    text="SEARCHING...",
                    fg=self.COLORS['warning']
                )
                self.current_status = DetectionStatus.SEARCHING
                
        else:  # SENTRY mode
            self.mag_display.configure(text=f"{result['magnitude']:.3f}")
            if result['alarm']:
                self.status_display.configure(
                    text="⚠️ WARNING:\nCOLLAPSE\nIMMINENT!",
                    fg=self.COLORS['danger']
                )
                self.current_status = DetectionStatus.COLLAPSE_WARNING
                # Flash effect
                self.root.configure(bg=self.COLORS['danger'])
                self.root.after(100, lambda: self.root.configure(bg=self.COLORS['bg_dark']))
            else:
                self.status_display.configure(
                    text="MONITORING...\nStable",
                    fg=self.COLORS['success']
                )
                
    def on_closing(self):
        """Handle window close event"""
        self.acquisition_thread.stop()
        time.sleep(0.2)  # Give thread time to stop
        self.root.destroy()
        
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for Life-Pulse v2.0"""
    print("=" * 60)
    print("  LIFE-PULSE v2.0 - Disaster Recovery System")
    print("  Dual-Mode Radar: Search (Heartbeat) / Sentry (Structure)")
    print("=" * 60)
    print("\nInitializing system...")
    print("  - Mock sensors enabled (no hardware required)")
    print("  - GUI launching...")
    print("\nControls:")
    print("  - Toggle 'Simulate Target' to inject test signals")
    print("  - Switch modes with the mode button")
    print("-" * 60)
    
    app = LifePulseGUI()
    app.run()


if __name__ == "__main__":
    main()
