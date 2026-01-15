"""
Life-Pulse v2.0 - Integrated Launcher
======================================
Launches Device Selection Screen first, then main monitoring application
"""

import tkinter as tk
from device_selection import DeviceSelectionScreen
from life_pulse_v2 import LifePulseGUI
import sys


class LifePulseLauncher:
    """Integrated launcher for Life-Pulse with device selection"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Life-Pulse v2.0 - Disaster Recovery System")
        self.root.geometry("1280x720")
        self.root.configure(bg='#1a1a2e')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.selected_device = None
        self.device_screen = None
        self.main_app = None
        self.current_frame = None
        
        # Show device selection screen
        self._show_device_selection()
    
    def _show_device_selection(self):
        """Display device selection screen"""
        print("[LAUNCHER] Showing device selection screen...")
        
        # Clear previous content
        if self.current_frame:
            self.current_frame.destroy()
        
        # Create container for device selection
        self.current_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.current_frame.pack(fill="both", expand=True)
        
        # Create device selection screen inside current frame
        self.device_screen = DeviceSelectionScreen(
            self.current_frame,
            on_device_selected=self._on_device_selected
        )
    
    def _on_device_selected(self, device_number):
        """Handle device selection"""
        print(f"[LAUNCHER] Device {device_number} selected")
        self.selected_device = device_number
        
        # Schedule transition to main app
        self.root.after(500, self._show_main_app)
    
    def _show_main_app(self):
        """Launch main Life-Pulse monitoring application"""
        print(f"[LAUNCHER] Launching main app for Device {self.selected_device}...")
        
        try:
            # Clear device selection screen
            if self.current_frame:
                self.current_frame.destroy()
            
            # Create container for main app
            self.current_frame = tk.Frame(self.root, bg='#1a1a2e')
            self.current_frame.pack(fill="both", expand=True)
            
            # Create main app with device info and custom root
            self.main_app = LifePulseGUI(
                selected_device=self.selected_device,
                parent_frame=self.current_frame,
                launcher=self
            )
            
        except Exception as e:
            print(f"[ERROR] Failed to launch main app: {e}")
            import traceback
            traceback.print_exc()
            # Go back to device selection on error
            self._show_device_selection()
    
    def on_closing(self):
        """Handle window closing"""
        try:
            if self.main_app and hasattr(self.main_app, 'acquisition_thread'):
                self.main_app.acquisition_thread.stop()
        except:
            pass
        
        try:
            self.root.destroy()
        except:
            pass
        
        sys.exit(0)
    
    def run(self):
        """Run the launcher"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
        except Exception as e:
            print(f"[LAUNCHER ERROR] {e}")
            import traceback
            traceback.print_exc()
            self.on_closing()


if __name__ == "__main__":
    print("=" * 60)
    print("  LIFE-PULSE v2.0 - INTEGRATED LAUNCHER")
    print("  Disaster Recovery Dual-Mode Radar System")
    print("=" * 60)
    
    launcher = LifePulseLauncher()
    launcher.run()
