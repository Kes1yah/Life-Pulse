"""
Life-Pulse v2.0 - Device Selection Screen
==========================================
Initial screen to select which device to monitor
"""

import tkinter as tk
from tkinter import ttk
import threading
import time


class DeviceSelectionScreen:
    """
    Device Selection Screen for Life-Pulse v2.0
    
    Features:
    - Select from 4 devices
    - Real-time notification display
    - Navigation to main monitoring screen
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
        'device_active': '#00ff88',
        'device_inactive': '#444466'
    }
    
    def __init__(self, parent, on_device_selected=None):
        """
        Initialize device selection screen
        
        Args:
            parent: Tkinter parent (root window or frame)
            on_device_selected: Callback function when device is selected
                              Should take device_number as parameter
        """
        self.parent = parent
        self.on_device_selected = on_device_selected
        self.selected_device = None
        self.notifications = []
        self.notification_queue = []  # Queue for notifications
        
        # Create widgets
        self._create_widgets()
        
    def _create_widgets(self):
        """Create all widgets for device selection screen"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=self.COLORS['bg_dark'])
        main_frame.pack(fill="both", expand=True)
        
        # === Header ===
        header_frame = tk.Frame(main_frame, bg=self.COLORS['bg_medium'], height=100)
        header_frame.pack(fill="x", padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🔴 LIFE-PULSE v2.0 - DEVICE SELECTION",
            font=("Helvetica", 28, "bold"),
            fg=self.COLORS['accent'],
            bg=self.COLORS['bg_medium']
        )
        title_label.pack(pady=20)
        
        subtitle = tk.Label(
            header_frame,
            text="Select a device to begin monitoring survivors",
            font=("Helvetica", 12),
            fg=self.COLORS['text_dim'],
            bg=self.COLORS['bg_medium']
        )
        subtitle.pack(pady=(0, 10))
        
        # === Main Content ===
        content_frame = tk.Frame(main_frame, bg=self.COLORS['bg_dark'])
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        
        # === Device Buttons (2x2 Grid) ===
        self.device_buttons = {}
        self.device_status_labels = {}
        
        for i in range(1, 5):
            row = (i - 1) // 2
            col = (i - 1) % 2
            
            # Device button frame
            device_frame = tk.Frame(
                content_frame,
                bg=self.COLORS['device_inactive'],
                relief="raised",
                bd=3,
                padx=20,
                pady=20
            )
            device_frame.grid(row=row, column=col, sticky="nsew", padx=15, pady=15)
            device_frame.grid_propagate(False)
            device_frame.configure(height=150, width=250)
            
            # Device number label
            device_label = tk.Label(
                device_frame,
                text=f"DEVICE {i}",
                font=("Helvetica", 24, "bold"),
                fg=self.COLORS['text'],
                bg=self.COLORS['device_inactive']
            )
            device_label.pack(pady=10)
            
            # Status label
            status_label = tk.Label(
                device_frame,
                text="● INACTIVE",
                font=("Helvetica", 12),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['device_inactive']
            )
            status_label.pack(pady=5)
            self.device_status_labels[i] = status_label
            
            # Select button
            def make_callback(device_num):
                return lambda: self._select_device(device_num)
            
            select_button = tk.Button(
                device_frame,
                text="SELECT",
                font=("Helvetica", 12, "bold"),
                fg=self.COLORS['bg_dark'],
                bg=self.COLORS['accent'],
                activebackground=self.COLORS['success'],
                relief="flat",
                command=make_callback(i),
                width=15,
                height=1
            )
            select_button.pack(pady=10)
            
            self.device_buttons[i] = {
                'frame': device_frame,
                'button': select_button,
                'label': device_label,
                'status': status_label
            }
        
        # === Notification Panel ===
        self._create_notification_panel(main_frame)
        
    def _create_notification_panel(self, parent):
        """Create notification panel at bottom"""
        notif_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'], height=120)
        notif_frame.pack(fill="x", padx=10, pady=10)
        notif_frame.pack_propagate(False)
        
        notif_title = tk.Label(
            notif_frame,
            text="📢 NOTIFICATIONS & ALERTS",
            font=("Helvetica", 12, "bold"),
            fg=self.COLORS['warning'],
            bg=self.COLORS['bg_medium']
        )
        notif_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Notification text area
        self.notif_text = tk.Text(
            notif_frame,
            height=4,
            font=("Courier", 9),
            fg=self.COLORS['success'],
            bg=self.COLORS['bg_dark'],
            state="disabled",
            relief="flat",
            padx=10,
            pady=5
        )
        self.notif_text.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
    def _select_device(self, device_number):
        """Handle device selection"""
        self.selected_device = device_number
        
        # Update UI
        for dev_num, buttons in self.device_buttons.items():
            if dev_num == device_number:
                buttons['frame'].configure(bg=self.COLORS['device_active'])
                buttons['label'].configure(bg=self.COLORS['device_active'])
                buttons['status'].configure(bg=self.COLORS['device_active'], fg=self.COLORS['success'])
                buttons['button'].configure(bg=self.COLORS['success'], text="✓ SELECTED")
            else:
                buttons['frame'].configure(bg=self.COLORS['device_inactive'])
                buttons['label'].configure(bg=self.COLORS['device_inactive'])
                buttons['status'].configure(bg=self.COLORS['device_inactive'], fg=self.COLORS['text_dim'])
                buttons['button'].configure(bg=self.COLORS['accent'], text="SELECT")
        
        # Add notification
        self._add_notification(f"✓ Device {device_number} selected for monitoring")
        
        # Callback to main application
        if self.on_device_selected:
            self.parent.winfo_toplevel().after(500, lambda: self.on_device_selected(device_number))
    
    def simulate_device_detection(self, device_number, person_name):
        """
        Simulate a detection from a device
        (In real scenario, this would come from hardware)
        
        Args:
            device_number: Which device detected
            person_name: Name of person detected
        """
        self._add_notification(f"🔔 ALERT: Device {device_number} detected {person_name}!")
        
        # Highlight the device that detected
        self.device_status_labels[device_number].configure(text="● DETECTING", fg=self.COLORS['success'])
        self.root.after(2000, lambda: self.device_status_labels[device_number].configure(
            text="● ACTIVE", fg=self.COLORS['text_dim']
        ))
    
    def _add_notification(self, message):
        """Add notification to queue"""
        self.notification_queue.append({
            'message': message,
            'timestamp': time.time()
        })
        
        # Keep only last 20 notifications
        if len(self.notification_queue) > 20:
            self.notification_queue.pop(0)
    
    def _update_notifications(self):
        """Update notification display"""
        self.notif_text.configure(state="normal")
        self.notif_text.delete("1.0", "end")
        
        # Display last 8 notifications
        display_notifs = self.notification_queue[-8:]
        for notif in display_notifs:
            self.notif_text.insert("end", f"{notif['message']}\n")
        
        self.notif_text.configure(state="disabled")
        self.notif_text.see("end")
        
        # Schedule next update
        try:
            self.parent.winfo_toplevel().after(500, self._update_notifications)
        except:
            pass


# Demo/Testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Life-Pulse v2.0 - Device Selection")
    root.geometry("1280x720")
    root.configure(bg='#1a1a2e')
    
    def on_device_selected(device_num):
        print(f"Device {device_num} selected!")
        # This would launch the main monitoring screen
        root.destroy()  # Close selection screen
        # Main application will launch after this
    
    screen = DeviceSelectionScreen(root, on_device_selected=on_device_selected)
    root.mainloop()
