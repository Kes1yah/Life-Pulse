"""
Life-Pulse v2.0 - Disaster Recovery Dual-Mode Radar System
===========================================================
A Raspberry Pi 5 based system for:
  - Mode A (Search): Detecting human heartbeats/breathing (0.2Hz - 2.0Hz)
  - Mode B (Sentry): Detecting structural vibrations/wall shifts (10Hz+)

Author: Embedded Systems Implementation
Hardware: Raspberry Pi 5 with radar sensor, MPU-6050, 5" touchscreen

Features:
  - Real-time signal processing with FFT
  - Survivor detection with photo display
  - Missing persons database integration
  - "FOUND" status tracking
  - WiFi image upload to backend for face recognition

Hardware Integration Note:
--------------------------
This code uses Mock sensor classes for testing. To integrate with real hardware:
1. Replace MockRadarSensor with actual radar interface (SPI/I2C/Serial)
2. Replace MockVibrationSensor with MPU-6050 I2C driver
3. Replace MockCamera with picamera2 or OpenCV camera capture
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import threading
import time
import os
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import queue
import requests
import json

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from io import BytesIO

# Local imports
from database import DisasterDatabase, MissingPerson, WebcamCamera
import pickle
import cv2
import numpy as np


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
    IDENTIFYING = "IDENTIFYING"
    MATCH_FOUND = "MATCH_FOUND"
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
    
    # Paths
    assets_dir: str = "assets"
    db_path: str = "life_pulse.db"


# =============================================================================
# MOCK SENSOR CLASSES (Replace with hardware drivers for production)
# =============================================================================

class MockRadarSensor:
    """
    Mock Radar Sensor for testing without hardware.
    
    Generates a noisy sine wave with optional target injection:
    - In SEARCH mode: Injects 1.2Hz sine wave (simulated heartbeat)
    - In SENTRY mode: Injects random high-amplitude spikes (structural shift)
    
    Hardware Integration:
    --------------------
    Replace this class with actual radar sensor driver:
    - For SPI radar: Use spidev library
    - For serial radar: Use pyserial
    - For I2C radar: Use smbus2 or board/busio
    
    Example:
    ```python
    import spidev
    
    class RealRadarSensor:
        def __init__(self, config):
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)  # Bus 0, Device 0
            self.spi.max_speed_hz = 1000000
            
        def read_samples(self, num_samples):
            # Read from actual ADC
            data = []
            for _ in range(num_samples):
                raw = self.spi.xfer2([0x00, 0x00])
                value = ((raw[0] & 0x0F) << 8) | raw[1]
                data.append(value / 4096.0)  # Normalize
            return np.array(data)
    ```
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
    
    Hardware Integration:
    --------------------
    Replace with actual MPU-6050 driver:
    ```python
    import smbus2
    
    class RealVibrationSensor:
        MPU6050_ADDR = 0x68
        ACCEL_XOUT_H = 0x3B
        
        def __init__(self, config):
            self.bus = smbus2.SMBus(1)  # I2C bus 1
            # Wake up MPU-6050
            self.bus.write_byte_data(self.MPU6050_ADDR, 0x6B, 0)
            
        def read_acceleration(self, num_samples):
            x_data, y_data, z_data = [], [], []
            for _ in range(num_samples):
                data = self.bus.read_i2c_block_data(self.MPU6050_ADDR, self.ACCEL_XOUT_H, 6)
                x = (data[0] << 8) | data[1]
                y = (data[2] << 8) | data[3]
                z = (data[4] << 8) | data[5]
                # Convert to G-force
                x_data.append(x / 16384.0)
                y_data.append(y / 16384.0)
                z_data.append(z / 16384.0)
            return np.array(x_data), np.array(y_data), np.array(z_data)
    ```
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
    - Survivor photo and details display
    - Database integration for missing persons
    - "FOUND" status tracking with green indicator
    - Dark theme interface
    - Device monitoring mode
    """
    
    # Color scheme (dark theme)
    COLORS = {
        'bg_dark': '#1a1a2e',
        'bg_medium': '#16213e',
        'bg_light': '#0f3460',
        'accent': '#e94560',
        'success': '#00ff88',
        'success_dark': '#00aa55',
        'warning': '#ffaa00',
        'danger': '#ff4444',
        'text': '#ffffff',
        'text_dim': '#888888',
        'graph_line': '#00ffff',
        'graph_grid': '#333355',
        'found_green': '#00ff00',
        'found_glow': '#66ff66'
    }
    
    def __init__(self, selected_device=None, parent_frame=None, launcher=None):
        self.config = SystemConfig()
        self.selected_device = selected_device  # Device number (1-4)
        self.launcher = launcher  # Reference to launcher for navigation
        
        # Initialize database
        self.database = DisasterDatabase(self.config.db_path)
        
        # Initialize sensors
        self.radar_sensor = MockRadarSensor(self.config)
        self.vibration_sensor = MockVibrationSensor(self.config)
        self.signal_processor = SignalProcessor(self.config)
        self.camera = WebcamCamera(self.config.assets_dir)
        
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
        self.current_person: Optional[MissingPerson] = None
        self.survivor_detected = False
        self.identifying_in_progress = False
        self.photo_image = None  # Keep reference to prevent garbage collection
        
        # Use provided parent frame or create new root window
        if parent_frame is not None:
            self.parent_frame = parent_frame
            self.root = parent_frame.winfo_toplevel()
        else:
            # Create main window (standalone mode)
            self.root = tk.Tk()
            self.root.title(f"Life-Pulse v2.0 - Device {selected_device} Monitoring" if selected_device else "Life-Pulse v2.0")
            self.root.geometry("1280x720")
            self.root.configure(bg=self.COLORS['bg_dark'])
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Make it responsive to touch/resize
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(0, weight=1)
            
            self.parent_frame = None
        
        self._create_widgets()
        self._load_next_missing_person()
        self._start_acquisition()
        
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        if self.parent_frame is not None:
            main_frame = self.parent_frame
        else:
            main_frame = tk.Frame(self.root, bg=self.COLORS['bg_dark'])
            main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # === Header Panel ===
        self._create_header(main_frame)
        
        # === Main Content Area ===
        content_frame = tk.Frame(main_frame, bg=self.COLORS['bg_dark'])
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=2)  # Image display
        content_frame.grid_columnconfigure(1, weight=1)  # Person details
        content_frame.grid_columnconfigure(2, weight=1)  # Status
        
        # === Image Display Panel (Left) ===
        self._create_image_display_panel(content_frame)
        
        # === Person Details Panel (Center) ===
        self._create_person_panel(content_frame)
        
        # === Status Panel (Right) ===
        self._create_status_panel(content_frame)
        
        # === Footer ===
        self._create_footer(main_frame)
        
    def _create_header(self, parent):
        """Create header panel"""
        header_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'], height=80)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Back button (if launcher available)
        if self.launcher:
            back_button = tk.Button(
                header_frame,
                text="◀ BACK",
                font=("Helvetica", 10, "bold"),
                fg=self.COLORS['bg_dark'],
                bg=self.COLORS['warning'],
                activebackground=self.COLORS['accent'],
                relief="flat",
                command=self._go_back_to_devices,
                padx=15,
                pady=5
            )
            back_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🔴 LIFE-PULSE v2.0",
            font=("Helvetica", 24, "bold"),
            fg=self.COLORS['accent'],
            bg=self.COLORS['bg_medium']
        )
        title_label.grid(row=0, column=1, padx=20, pady=10, sticky="w")
        
        # Subtitle with device info
        device_info = f"Device {self.selected_device} Monitoring" if self.selected_device else "Disaster Recovery Radar"
        subtitle = tk.Label(
            header_frame,
            text=device_info,
            font=("Helvetica", 10),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_medium']
        )
        subtitle.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="w")
        
        # Database stats
        stats = self.database.get_statistics()
        self.stats_label = tk.Label(
            header_frame,
            text=f"Missing: {stats['missing']} | Found: {stats['found']}",
            font=("Helvetica", 11),
            fg=self.COLORS['warning'],
            bg=self.COLORS['bg_medium']
        )
        self.stats_label.grid(row=0, column=1, rowspan=2, padx=20)
        
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
        
    def _create_image_display_panel(self, parent):
        """Create graph panel for bio-signals (noise/heart rate)"""
        graph_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = tk.Label(
            graph_frame,
            text="VITAL SIGNALS MONITOR",
            font=("Helvetica", 11, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium'],
            padx=10,
            pady=5
        )
        title_label.pack(side=tk.TOP)
        
        # Graph container
        graph_container = tk.Frame(graph_frame, bg=self.COLORS['bg_light'], relief=tk.SUNKEN, bd=2)
        graph_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        graph_container.grid_rowconfigure(0, weight=1)
        graph_container.grid_columnconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor=self.COLORS['bg_medium'])
        self.ax = self.fig.add_subplot(111)
        self._style_plot()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Initialize plot line
        self.line, = self.ax.plot([], [], color=self.COLORS['graph_line'], linewidth=1)
        
        # Signal label
        self.signal_label = tk.Label(
            graph_frame,
            text="Status: Waiting for signal...",
            font=("Helvetica", 9),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_medium']
        )
        self.signal_label.pack(pady=(5, 10))
        
    def _create_person_panel(self, parent):
        """Create person details panel with captured image"""
        person_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        person_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Title
        tk.Label(
            person_frame,
            text="HARDWARE IMAGE FEED",
            font=("Helvetica", 12, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium']
        ).pack(pady=(15, 10))
        
        # Image frame - for captured hardware image
        image_container = tk.Frame(person_frame, bg=self.COLORS['bg_light'], padx=5, pady=5, relief=tk.SUNKEN, bd=2)
        image_container.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.photo_label = tk.Label(
            image_container,
            text="Waiting for hardware image...",
            font=("Helvetica", 10),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_light'],
            width=20,
            height=12
        )
        self.photo_label.pack()
        
        # Person details frame
        details_frame = tk.Frame(person_frame, bg=self.COLORS['bg_medium'])
        details_frame.pack(padx=10, pady=10, fill="x")
        
        # Name
        self.name_label = tk.Label(
            details_frame,
            text="Name: --",
            font=("Helvetica", 14, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium'],
            anchor="center"
        )
        self.name_label.pack(fill="x", pady=10)
        
        # Status indicator
        self.person_status_label = tk.Label(
            details_frame,
            text="STATUS: MISSING",
            font=("Helvetica", 11, "bold"),
            fg=self.COLORS['danger'],
            bg=self.COLORS['bg_medium'],
            anchor="center"
        )
        self.person_status_label.pack(fill="x", pady=(0, 5))
        
        # === FOUND Button (Green Light Indicator) ===
        self.found_button = tk.Button(
            person_frame,
            text="✓ MARK AS FOUND",
            font=("Helvetica", 14, "bold"),
            fg=self.COLORS['bg_dark'],
            bg=self.COLORS['text_dim'],
            activebackground=self.COLORS['found_green'],
            activeforeground=self.COLORS['bg_dark'],
            relief="flat",
            command=self.mark_as_found,
            width=18,
            height=2,
            state="disabled"
        )
        self.found_button.pack(pady=15)
        
        # Next person button
        self.next_button = tk.Button(
            person_frame,
            text="NEXT PERSON ▶",
            font=("Helvetica", 10),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_light'],
            activebackground=self.COLORS['accent'],
            relief="flat",
            command=self._load_next_missing_person,
            width=15
        )
        self.next_button.pack(pady=5)
        
        # Enrollment button
        self.enroll_button = tk.Button(
            person_frame,
            text="👤 ENROLL SURVIVOR",
            font=("Helvetica", 10, "bold"),
            fg=self.COLORS['bg_dark'],
            bg=self.COLORS['warning'],
            activebackground=self.COLORS['accent'],
            relief="flat",
            command=self._enroll_faces_dialog,
            width=18
        )
        self.enroll_button.pack(pady=(20, 5))
        
    def _create_status_panel(self, parent):
        """Create status panel"""
        status_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        status_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        
        # Status title
        tk.Label(
            status_frame,
            text="SYSTEM STATUS",
            font=("Helvetica", 12, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium']
        ).pack(pady=(20, 10))
        
        # Main status display
        self.status_display = tk.Label(
            status_frame,
            text="SEARCHING...",
            font=("Helvetica", 16, "bold"),
            fg=self.COLORS['warning'],
            bg=self.COLORS['bg_dark'],
            wraplength=180,
            justify="center",
            padx=15,
            pady=25
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
            font=("Helvetica", 18, "bold"),
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
            font=("Helvetica", 18, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_medium']
        )
        self.mag_display.pack()
        
        # Separator
        ttk.Separator(status_frame, orient="horizontal").pack(fill="x", padx=10, pady=15)
        
        # Stability Status (for SENTRY mode)
        stability_frame = tk.Frame(status_frame, bg=self.COLORS['bg_medium'])
        stability_frame.pack(padx=10, pady=10, fill="x")
        
        tk.Label(
            stability_frame,
            text="Structural Status:",
            font=("Helvetica", 10),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_medium']
        ).pack()
        
        self.stability_display = tk.Label(
            stability_frame,
            text="STABLE",
            font=("Helvetica", 14, "bold"),
            fg=self.COLORS['success'],
            bg=self.COLORS['bg_medium']
        )
        self.stability_display.pack()
        
        # Separator
        ttk.Separator(status_frame, orient="horizontal").pack(fill="x", padx=10, pady=15)
        
        # Mode toggle button
        self.mode_button = tk.Button(
            status_frame,
            text="SWITCH TO\nSENTRY MODE",
            font=("Helvetica", 11, "bold"),
            fg=self.COLORS['text'],
            bg=self.COLORS['bg_light'],
            activebackground=self.COLORS['accent'],
            activeforeground=self.COLORS['text'],
            relief="flat",
            command=self.toggle_mode,
            width=14,
            height=2
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
        
        # Reset database button (for testing)
        reset_btn = tk.Button(
            status_frame,
            text="Reset All to Missing",
            font=("Helvetica", 9),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_dark'],
            relief="flat",
            command=self._reset_database
        )
        reset_btn.pack(pady=5)
        
    def _create_footer(self, parent):
        """Create footer panel"""
        footer_frame = tk.Frame(parent, bg=self.COLORS['bg_dark'], height=30)
        footer_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        footer_label = tk.Label(
            footer_frame,
            text="Life-Pulse",
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
        
    def _display_captured_image(self, photo_path: str):
        """Display captured image in the hardware image feed area"""
        try:
            captured_frame = None
            
            # Get image from memory or disk
            if photo_path == "memory://captured_image":
                captured_frame = self.camera.get_last_captured_frame()
            else:
                if os.path.exists(photo_path):
                    captured_frame = cv2.imread(photo_path)
            
            if captured_frame is not None:
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display in image_label (approx 400x300)
                frame_resized = cv2.resize(frame_rgb, (400, 300))
                
                # Convert to PIL Image
                img = Image.fromarray(frame_resized)
                
                # Convert to PhotoImage
                self.captured_image = ImageTk.PhotoImage(img)
                
                # Display in hardware image feed area
                self.image_label.configure(image=self.captured_image, text="")
                self.image_label.image = self.captured_image
                
        except Exception as e:
            print(f"Error displaying captured image: {e}")
            self.image_label.configure(text=f"Image Error: {str(e)[:30]}")
    
    def _load_photo(self, photo_path: str):
        """Load and display person photo"""
        try:
            if os.path.exists(photo_path):
                img = Image.open(photo_path)
                img = img.resize((120, 120), Image.Resampling.LANCZOS)
                self.photo_image = ImageTk.PhotoImage(img)
                self.photo_label.configure(image=self.photo_image, text="", width=120, height=120)
            else:
                self.photo_label.configure(image="", text="No Photo", width=15, height=8)
                self.photo_image = None
        except Exception as e:
            print(f"Error loading photo: {e}")
            self.photo_label.configure(image="", text="Photo Error", width=15, height=8)
            self.photo_image = None
            
    def _load_next_missing_person(self):
        """Load next missing person from database (Cycling display)"""
        current_id = self.current_person.id if self.current_person else 0
        self.current_person = self.database.get_next_missing(current_id)
        
        if self.current_person:
            # Update UI with person name
            self.name_label.configure(text=f"Name: {self.current_person.name}")
            self.person_status_label.configure(text="STATUS: MISSING", fg=self.COLORS['danger'])
            
            # Load photo (if it exists)
            self._load_photo(self.current_person.photo_path)
            
            # Reset found button state
            self.found_button.configure(
                state="disabled",
                bg=self.COLORS['text_dim'],
                fg=self.COLORS['bg_dark']
            )
        else:
            # No more missing persons
            self.name_label.configure(text="Name: All Found!")
            self.person_status_label.configure(text="ALL CLEAR", fg=self.COLORS['success'])
            self.photo_label.configure(image="", text="✓", width=15, height=8)
            self.photo_image = None
            
        self._update_stats()
        
    def _update_stats(self):
        """Update database statistics display"""
        stats = self.database.get_statistics()
        self.stats_label.configure(text=f"Missing: {stats['missing']} | Found: {stats['found']}")
        
    def _reset_database(self):
        """Reset all persons to missing status (for testing)"""
        self.database.reset_all_to_missing()
        self.current_person = None
        self._load_next_missing_person()
        messagebox.showinfo("Reset Complete", "All persons reset to MISSING status.")
        
    def mark_as_found(self):
        """Mark current person as found in database"""
        if self.current_person:
            success = self.database.mark_as_found(
                self.current_person.id,
                "Located by Life-Pulse radar system"
            )
            
            if success:
                # Update UI
                self.person_status_label.configure(text="STATUS: FOUND ✓", fg=self.COLORS['found_green'])
                self.found_button.configure(
                    text="✓ MARKED FOUND",
                    state="disabled",
                    bg=self.COLORS['success_dark']
                )
                
                # Flash green effect
                self._flash_found_indicator()
                
                # Show confirmation
                messagebox.showinfo(
                    "Survivor Located",
                    f"{self.current_person.name} has been marked as FOUND!\n\n"
                    "The central registry has been updated."
                )
                
                # Load next person after short delay
                self.root.after(1000, self._load_next_missing_person)
                
            self._update_stats()
            
    def _flash_found_indicator(self):
        """Flash green indicator for found status"""
        original_bg = self.root.cget('bg')
        
        def flash(count=0):
            if count < 6:
                if count % 2 == 0:
                    self.root.configure(bg=self.COLORS['found_green'])
                    self.found_button.configure(bg=self.COLORS['found_glow'])
                else:
                    self.root.configure(bg=original_bg)
                    self.found_button.configure(bg=self.COLORS['success_dark'])
                self.root.after(150, lambda: flash(count + 1))
            else:
                self.root.configure(bg=original_bg)
                
        flash()
        
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
                
            # Update plot based on human detection status
            if self.survivor_detected:
                # Show wave movements
                self.signal_label.configure(
                    text="Status: HUMAN DETECTED",
                    fg=self.COLORS['success']
                )
            else:
                # Show noise
                self.signal_label.configure(
                    text="Status: Ambient Noise (No Human)",
                    fg=self.COLORS['text_dim']
                )
            
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
            # Check for heartbeat/respiration detection
            if result['detected']:
                if not self.survivor_detected and not self.identifying_in_progress:
                    # New detection! Trigger staged identification flow
                    self._start_staged_identification()
                
                # Update realtime displays
                self.freq_display.configure(text=f"{result['frequency']:.2f} Hz")
                self.mag_display.configure(text=f"{result['magnitude']:.3f}")
            else:
                # Signal lost
                if not self.identifying_in_progress:
                    self.survivor_detected = False
                    self.freq_display.configure(text="-- Hz")
                    self.mag_display.configure(text=f"{result['magnitude']:.3f}")
                    self._reset_identification_ui()
                    
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

    def _start_staged_identification(self):
        """Stage 1: Survivor Detected & Image Capture"""
        self.survivor_detected = True
        self.identifying_in_progress = True
        self.current_status = DetectionStatus.SURVIVOR_DETECTED
        
        self.status_display.configure(
            text="🚨 SURVIVOR DETECTED!\nCAPTURING IMAGE...",
            fg=self.COLORS['accent']
        )
        
        # Trigger "Hardware" capture (keep in memory, don't save to assets folder)
        captured_path = self.camera.capture(save_to_disk=False)
        
        if captured_path:
            # Move to Stage 2 after short delay
            # During this delay, we could do more processing if needed
            self.root.after(3000, lambda: self._identifying_phase(captured_path))
        else:
            self.status_display.configure(
                text="❌ CAMERA ERROR\nCHECK HARDWARE",
                fg=self.COLORS['danger']
            )
            self.identifying_in_progress = False

    def _identifying_phase(self, photo_path: str):
        """Stage 2: Analysis / Cross-Verification - Send image to backend"""
        self.current_status = DetectionStatus.IDENTIFYING
        
        # Display captured image in hardware feed area
        self._display_captured_image(photo_path)
        
        self.status_display.configure(
            text="📤 UPLOADING...\nTO BACKEND...",
            fg=self.COLORS['warning']
        )
        
        # Send image to backend for verification (non-blocking)
        self.root.after(500, lambda: self._send_image_to_backend(photo_path))

    def _send_image_to_backend(self, photo_path: str):
        """Send captured image to backend server for face recognition"""
        try:
            backend_url = "http://192.168.202.227:5000/api/image-upload"  # Backend IP
            
            # Get image from memory or disk
            if photo_path == "memory://captured_image":
                captured_frame = self.camera.get_last_captured_frame()
                if captured_frame is not None:
                    _, buffer = cv2.imencode('.jpg', captured_frame)
                    image_bytes = buffer.tobytes()
                else:
                    self.status_display.configure(
                        text="❌ NO IMAGE\nTO SEND",
                        fg=self.COLORS['danger']
                    )
                    return
            else:
                with open(photo_path, 'rb') as f:
                    image_bytes = f.read()
            
            # Send to backend
            files = {'image': ('captured_image.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(backend_url, files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[CLIENT] Backend response: {result}")
                
                # Update UI based on response
                if result.get('human_presence'):
                    self.root.after(0, lambda: self._process_human_detected(result))
                else:
                    self.root.after(0, lambda: self._process_no_human_detected())
            else:
                self.status_display.configure(
                    text="⚠️  BACKEND\nERROR",
                    fg=self.COLORS['danger']
                )
                
        except requests.exceptions.ConnectionError:
            print("[CLIENT] ❌ Cannot connect to backend. Using local recognition...")
            self.status_display.configure(
                text="🔍 LOCAL\nVERIFICATION...",
                fg=self.COLORS['warning']
            )
            self.root.after(2000, lambda: self._complete_identification(photo_path))
            
        except Exception as e:
            print(f"[CLIENT] Error sending image: {e}")
            self.status_display.configure(
                text=f"❌ ERROR:\n{str(e)[:20]}",
                fg=self.COLORS['danger']
            )
    
    def update_sentry_stability(self, compromised: bool):
        """Update sentry mode structural integrity status"""
        if compromised:
            self.stability_display.configure(
                text="⚠️  COMPROMISED",
                fg=self.COLORS['danger']
            )
            self.status_display.configure(
                text="🚨 STRUCTURAL\nINTEGRITY\nCOMPROMISED!",
                fg=self.COLORS['danger']
            )
            # Flash warning
            self.root.configure(bg=self.COLORS['danger'])
            self.root.after(200, lambda: self.root.configure(bg=self.COLORS['bg_dark']))
        else:
            self.stability_display.configure(
                text="✅ STABLE",
                fg=self.COLORS['success']
            )
            self.status_display.configure(
                text="MONITORING...\nStable",
                fg=self.COLORS['success']
            )
    
    def _process_human_detected(self, backend_result: dict):
        """Process successful human detection from backend"""
        print(f"[CLIENT] ✅ HUMAN DETECTED from backend")
        
        # Switch graph to heart rate display
        self.survivor_detected = True
        self.signal_label.configure(
            text="🫀 Heart Rate Detected (from Backend)",
            fg=self.COLORS['success']
        )
        
        # Check if matched
        matched_name = backend_result.get('matched_person')
        if matched_name:
            self.status_display.configure(
                text=f"✅ MATCH FOUND!\n{matched_name}",
                fg=self.COLORS['success']
            )
        else:
            self.status_display.configure(
                text="✅ HUMAN DETECTED\n(Not in Database)",
                fg=self.COLORS['warning']
            )
            matched_name = "Unknown Person"
        
        self.root.after(3000, lambda: self._complete_identification_with_data(backend_result))
    
    def _process_no_human_detected(self):
        """Process no human detected from backend"""
        print("[CLIENT] ⚠️  No human detected")
        self.status_display.configure(
            text="❌ NO HUMAN\nDETECTED",
            fg=self.COLORS['danger']
        )
        self.survivor_detected = False
        self.identifying_in_progress = False
        self.root.after(3000, lambda: self._reset_identification_ui())

    def _complete_identification_with_data(self, backend_result: dict):
        """Stage 3: Complete identification with backend data"""
        if not self.survivor_detected:
            self.identifying_in_progress = False
            self._reset_identification_ui()
            return
        
        matched_name = backend_result.get('matched_person')
        person_id = backend_result.get('person_id')
        
        if matched_name and person_id:
            # Get person from database
            matched_person = self.database.get_person_by_id(person_id)
            if matched_person:
                self.current_person = matched_person
                self.current_status = DetectionStatus.MATCH_FOUND
                
                # Update GUI
                self.name_label.configure(text=f"Name: {matched_name}")
                self.person_status_label.configure(
                    text="STATUS: FOUND (VERIFIED)",
                    fg=self.COLORS['found_green']
                )
                
                # Enable FOUND button
                self.found_button.configure(
                    state="normal",
                    bg=self.COLORS['found_green'],
                    fg=self.COLORS['bg_dark']
                )
        
        self.identifying_in_progress = False
    
    def _complete_identification(self, photo_path: str):
        """Stage 3: Match Found & Display"""
        if not self.survivor_detected: # Check if signal still present
            self.identifying_in_progress = False
            self._reset_identification_ui()
            return
            
        # Map captured photo to database record using face-recognition
        matched_person = self._find_face_match(photo_path)
        
        if matched_person:
            self.current_person = matched_person
            self.current_status = DetectionStatus.MATCH_FOUND
            
            # Update GUI with matched name
            self.name_label.configure(text=f"Name: {matched_person.name}")
            self.person_status_label.configure(text="STATUS: FOUND (MATCHED)", fg=self.COLORS['found_green'])
            
            # Show photo
            self._load_photo(photo_path) # Show the live capture
            
            # Update main status display
            self.status_display.configure(
                text=f"✅ MATCH FOUND!\n{matched_person.name}",
                fg=self.COLORS['success']
            )
            
            # Enable FOUND button
            self.found_button.configure(
                state="normal",
                bg=self.COLORS['found_green'],
                fg=self.COLORS['bg_dark']
            )
            
            # Mark in DB automatically
            self.database.mark_as_found(matched_person.id, f"Automatically identified via webcam at {time.ctime()}")
            self._update_stats()
        else:
            self.status_display.configure(
                text="❌ UNKNOWN SURVIVOR\nNO MATCH FOUND",
                fg=self.COLORS['danger']
            )
            self._load_photo(photo_path) # Show the captured image anyway
            
        self.identifying_in_progress = False

    def _find_face_match(self, captured_photo_path: str) -> Optional[MissingPerson]:
        """Perform real face recognition matching against the database using OpenCV LBPH"""
        try:
            # Create recognizer and load data
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Get all persons with their multiple encodings
            entries = self.database.get_all_with_encodings()
            if not entries:
                return None
                
            faces = []
            labels = []
            id_map = {} # Internal ID to Person mapping
            
            label_idx = 0
            for entry in entries:
                person = entry['person']
                encodings = entry['encodings']
                
                for encoding_blob in encodings:
                    # Encoding is the face image bytes for LBPH
                    nparr = np.frombuffer(encoding_blob, np.uint8)
                    face_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if face_img is not None:
                        faces.append(face_img)
                        labels.append(label_idx)
                        id_map[label_idx] = person
                
                label_idx += 1
            
            if not faces:
                return None
                
            recognizer.train(faces, np.array(labels))
            
            # Detect face in capture - handle both file path and in-memory image
            if captured_photo_path == "memory://captured_image":
                # Get image from camera memory
                captured_img = self.camera.get_last_captured_frame()
            else:
                # Read from file path
                captured_img = cv2.imread(captured_photo_path)
            
            if captured_img is None:
                return None
                
            gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray) # Better contrast
            
            # Stricter face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 6)
            
            if len(detected_faces) == 0:
                print("DEBUG RECOGNITION: No face detected in capture.")
                return None
                
            x, y, w, h = detected_faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200)) # Standardize size
            
            label_id, confidence = recognizer.predict(roi_gray)
            matched_person = id_map.get(label_id)
            name = matched_person.name if matched_person else "UNKNOWN"
            
            print(f"DEBUG RECOGNITION: Matched {name} with confidence (distance): {confidence:.2f}")
            
            # Confidence for LBPH is distance (lower is better)
            # 100 is generally a safe upper bound for a positive match in varied lighting
            if confidence < 100: 
                return matched_person
                
            return None
            
        except Exception as e:
            print(f"Recognition Error: {e}")
            return None

    def _reset_identification_ui(self):
        """Clear the identification panel when no survivor is detected"""
        if not self.identifying_in_progress:
            self.status_display.configure(text="SEARCHING...", fg=self.COLORS['warning'])
            self.name_label.configure(text="Name: --")
            self.person_status_label.configure(text="STATUS: MISSING", fg=self.COLORS['danger'])
            self.photo_label.configure(image="", text="No Photo", width=15, height=8)
            self.photo_image = None
            self.found_button.configure(state="disabled", bg=self.COLORS['text_dim'])
                
    def _go_back_to_devices(self):
        """Go back to device selection screen"""
        if self.launcher:
            print("[APP] Going back to device selection...")
            # Stop acquisition
            self.acquisition_thread.stop()
            time.sleep(0.2)
            # Destroy all widgets in parent frame
            if self.parent_frame:
                for widget in self.parent_frame.winfo_children():
                    widget.destroy()
            # Show device selection
            self.launcher._show_device_selection()
        else:
            # If no launcher, just close the window
            self.on_closing()

    def on_closing(self):
        """Handle window close event"""
        self.acquisition_thread.stop()
        time.sleep(0.2)  # Give thread time to stop
        if not self.parent_frame:
            self.root.destroy()
        
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

    def app_run(self):
        """Run the main event loop"""
        self.root.mainloop()

    def _enroll_faces_dialog(self):
        """Show dialog to enroll faces for names in the database"""
        enroll_win = tk.Toplevel(self.root)
        enroll_win.title("Survivor Face Enrollment")
        enroll_win.geometry("400x300")
        enroll_win.configure(bg=self.COLORS['bg_medium'])
        
        tk.Label(
            enroll_win, 
            text="Enroll Face for Survivor",
            font=("Helvetica", 12, "bold"),
            bg=self.COLORS['bg_medium'],
            fg=self.COLORS['text']
        ).pack(pady=10)
        
        tk.Label(
            enroll_win,
            text="Select name and look at webcam:",
            bg=self.COLORS['bg_medium'],
            fg=self.COLORS['text_dim']
        ).pack()
        
        # Get missing persons
        missing = self.database.get_all_missing()
        if not missing:
            tk.Label(enroll_win, text="No survivors to enroll!", fg=self.COLORS['danger']).pack()
            return

        name_var = tk.StringVar(enroll_win)
        name_var.set(missing[0].name if missing else "")
        
        dropdown = ttk.Combobox(enroll_win, textvariable=name_var, state="readonly")
        dropdown['values'] = [p.name for p in missing]
        dropdown.pack(pady=20)

        def do_capture():
            selected_name = name_var.get()
            person = next((p for p in missing if p.name == selected_name), None)
            
            if person:
                self.status_display.configure(text="CAPTURING...", fg=self.COLORS['accent'])
                
                # Take photo
                save_path = os.path.join(self.config.assets_dir, f"ref_{person.name.lower()}_{int(time.time())}.png")
                photo_path = self.camera.capture(save_path)
                
                if photo_path:
                    try:
                        # Extract Face ROI for LBPH
                        img = cv2.imread(photo_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray) # Standardize contrast
                        
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 6)
                        
                        if len(detected_faces) > 0:
                            x, y, w, h = detected_faces[0]
                            roi_gray = gray[y:y+h, x:x+w]
                            roi_gray = cv2.resize(roi_gray, (200, 200)) # Standardize
                            
                            # Encode to bytes
                            _, buffer = cv2.imencode('.png', roi_gray)
                            encoding_blob = buffer.tobytes()
                            
                            # Save to DB
                            self.database.update_face_encoding(person.id, encoding_blob, photo_path)
                            messagebox.showinfo("Success", f"Face enrolled for {person.name}!")
                        else:
                            messagebox.showerror("Error", "No face detected in capture. Please ensure clear lighting.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Enrollment Error: {e}")
                
                self.status_display.configure(text="SEARCHING...", fg=self.COLORS['warning'])

        tk.Button(
            enroll_win,
            text="📸 CAPTURE & ENROLL",
            bg=self.COLORS['success'],
            fg=self.COLORS['bg_dark'],
            command=do_capture
        ).pack(pady=20)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for Life-Pulse v2.0"""
    print("=" * 60)
    print("  LIFE-PULSE v2.0 - Disaster Recovery System")
    print("  Dual-Mode Radar: Search (Heartbeat) / Sentry (Structure)")
    print("=" * 60)
    print("\nInitializing system with Real Webcam + Face Recognition...")
    
    # Ensure Pillow is available
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("\nERROR: Pillow library required. Install with: pip install Pillow")
        return
    
    app = LifePulseGUI()
    app.app_run()

if __name__ == "__main__":
    main()
