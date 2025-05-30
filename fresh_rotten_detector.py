import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# Set the appearance mode and theme
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")  

IMAGE_SIZE = 224
FRESHNESS_MODEL_PATH = "fruit_freshness_model_best.h5"
FOOD_TYPE_MODEL_PATH = "food_type_model.h5"  # This would be your second model

# Nutritional values database for the supported fruits and vegetables
NUTRITIONAL_VALUES = {
    "Apple": {
        "Calories": "52 kcal", 
        "Carbs": "14g", 
        "Fiber": "2.4g", 
        "Sugar": "10g",
        "Protein": "0.3g",
        "Fat": "0.2g",
        "Vitamin C": "8% DV",
        "Potassium": "3% DV"
    },
    "Banana": {
        "Calories": "89 kcal",
        "Carbs": "23g",
        "Fiber": "2.6g",
        "Sugar": "12g",
        "Protein": "1.1g",
        "Fat": "0.3g",
        "Vitamin B6": "20% DV",
        "Potassium": "10% DV"
    },
    "Mango": {
        "Calories": "60 kcal",
        "Carbs": "15g",
        "Fiber": "1.6g",
        "Sugar": "14g",
        "Protein": "0.8g",
        "Fat": "0.4g",
        "Vitamin C": "60% DV",
        "Vitamin A": "8% DV"
    },
    "Orange": {
        "Calories": "47 kcal",
        "Carbs": "12g",
        "Fiber": "2.4g",
        "Sugar": "9g",
        "Protein": "0.9g",
        "Fat": "0.1g",
        "Vitamin C": "90% DV",
        "Folate": "10% DV"
    },
    "Strawberry": {
        "Calories": "32 kcal",
        "Carbs": "7.7g",
        "Fiber": "2g",
        "Sugar": "4.9g",
        "Protein": "0.7g",
        "Fat": "0.3g",
        "Vitamin C": "98% DV",
        "Manganese": "19% DV"
    },
    "BellPepper": {
        "Calories": "31 kcal",
        "Carbs": "6g",
        "Fiber": "2.1g",
        "Sugar": "4.2g",
        "Protein": "1g",
        "Fat": "0.3g",
        "Vitamin C": "169% DV",
        "Vitamin A": "7% DV"
    },
    "Bittergourd": {
        "Calories": "17 kcal",
        "Carbs": "3.7g",
        "Fiber": "2.8g",
        "Sugar": "1.8g",
        "Protein": "1g",
        "Fat": "0.2g",
        "Vitamin C": "84% DV",
        "Vitamin A": "44% DV"
    },
    "Capsicum": {
        "Calories": "31 kcal",
        "Carbs": "6g",
        "Fiber": "2.1g",
        "Sugar": "4.2g",
        "Protein": "1g",
        "Fat": "0.3g",
        "Vitamin C": "169% DV",
        "Vitamin A": "7% DV"
    },
    "Carrot": {
        "Calories": "41 kcal",
        "Carbs": "10g",
        "Fiber": "2.8g",
        "Sugar": "4.7g",
        "Protein": "0.9g",
        "Fat": "0.2g",
        "Vitamin A": "334% DV",
        "Vitamin K": "13% DV"
    },
    "Cucumber": {
        "Calories": "15 kcal",
        "Carbs": "3.6g",
        "Fiber": "0.5g",
        "Sugar": "1.7g",
        "Protein": "0.7g",
        "Fat": "0.1g",
        "Vitamin K": "16% DV",
        "Potassium": "4% DV"
    },
    "Okra": {
        "Calories": "33 kcal",
        "Carbs": "7g",
        "Fiber": "3.2g",
        "Sugar": "1.5g",
        "Protein": "1.9g",
        "Fat": "0.2g",
        "Vitamin C": "38% DV",
        "Folate": "22% DV"
    },
    "Potato": {
        "Calories": "77 kcal",
        "Carbs": "17g",
        "Fiber": "2.2g",
        "Sugar": "0.8g",
        "Protein": "2g",
        "Fat": "0.1g",
        "Vitamin C": "28% DV",
        "Potassium": "12% DV"
    },
    "Tomato": {
        "Calories": "18 kcal",
        "Carbs": "3.9g",
        "Fiber": "1.2g",
        "Sugar": "2.6g",
        "Protein": "0.9g",
        "Fat": "0.2g",
        "Vitamin C": "28% DV",
        "Potassium": "7% DV"
    }
}

class FoodClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fresh or Rotten Food Detector")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Model and class variables
        self.freshness_model = None
        self.food_type_model = None
        self.current_image_path = None
        self.current_image = None
        self.food_types = ["Apple", "Banana", "Mango", "Orange", "Strawberry", "BellPepper", 
                          "Bittergourd", "Capsicum", "Carrot", "Cucumber", "Okra", "Potato", "Tomato"]

        # Create UI components
        self.create_ui()
        
        # Load models
        self.load_models()
        
    def create_ui(self):
        # Configure the grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Main container
        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.main_container.grid_columnconfigure(0, weight=3)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        
        # Left panel (image display and upload)
        self.left_panel = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_panel.grid_rowconfigure(0, weight=0)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_frame = ctk.CTkFrame(self.left_panel, corner_radius=10, fg_color="#262626", height=60)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.header_frame.grid_propagate(False)
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=0)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="Food Quality Analyzer", 
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#1E90FF"
        )
        self.title_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")
        
        self.upload_button = ctk.CTkButton(
            self.header_frame, 
            text="Upload Image", 
            command=self.upload_image,
            fg_color="#1E90FF",
            hover_color="#0078D7",
            height=36
        )
        self.upload_button.grid(row=0, column=1, padx=15, pady=10)
        
        # Image display area
        self.image_frame = ctk.CTkFrame(self.left_panel, corner_radius=10, fg_color="#1A1A1A")
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)
        
        self.upload_placeholder = ctk.CTkFrame(self.image_frame, fg_color="#262626", corner_radius=10)
        self.upload_placeholder.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.8)
        self.upload_placeholder.grid_columnconfigure(0, weight=1)
        self.upload_placeholder.grid_rowconfigure(0, weight=1)
        self.upload_placeholder.grid_rowconfigure(1, weight=0)
        self.upload_placeholder.grid_rowconfigure(2, weight=0)
        
        self.upload_icon = ctk.CTkLabel(
            self.upload_placeholder, 
            text="📷", 
            font=ctk.CTkFont(size=48),
            text_color="#555555"
        )
        self.upload_icon.grid(row=0, column=0, padx=20, pady=(20, 5))
        
        self.upload_text = ctk.CTkLabel(
            self.upload_placeholder, 
            text="Drop image here or", 
            font=ctk.CTkFont(size=16),
            text_color="#666666"
        )
        self.upload_text.grid(row=1, column=0, padx=20)
        
        self.select_file_button = ctk.CTkButton(
            self.upload_placeholder, 
            text="Select File", 
            command=self.upload_image,
            fg_color="#333333",
            hover_color="#444444",
            height=32,
            width=120
        )
        self.select_file_button.grid(row=2, column=0, padx=20, pady=(5, 20))
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Right panel (results)
        self.right_panel = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_rowconfigure(0, weight=0)
        self.right_panel.grid_rowconfigure(1, weight=0)
        self.right_panel.grid_rowconfigure(2, weight=1)
        self.right_panel.grid_rowconfigure(3, weight=0)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # Results header
        self.result_header = ctk.CTkFrame(self.right_panel, corner_radius=10, fg_color="#262626", height=60)
        self.result_header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.result_header.grid_propagate(False)
        
        self.result_title = ctk.CTkLabel(
            self.result_header, 
            text="Results", 
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#1E90FF"
        )
        self.result_title.place(relx=0.5, rely=0.5, anchor="center")
        
        # Detection results
        self.detection_frame = ctk.CTkFrame(self.right_panel, corner_radius=10, fg_color="#1A1A1A")
        self.detection_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        self.food_type_label = ctk.CTkLabel(
            self.detection_frame, 
            text="Food Type: Awaiting Image", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.food_type_label.pack(padx=15, pady=(15, 5))
        
        self.freshness_label = ctk.CTkLabel(
            self.detection_frame, 
            text="Quality: Awaiting Analysis", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.freshness_label.pack(padx=15, pady=(5, 15))
        
        # Gauge chart
        self.gauge_frame = ctk.CTkFrame(self.right_panel, corner_radius=10, fg_color="#1A1A1A")
        self.gauge_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        self.fig = Figure(figsize=(4, 4), dpi=100, facecolor="#1A1A1A")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.gauge_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.draw_gauge(0, "Awaiting Analysis")
        
        # Nutrition info
        self.nutrition_frame = ctk.CTkFrame(self.right_panel, corner_radius=10, fg_color="#1A1A1A")
        self.nutrition_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        self.nutrition_title = ctk.CTkLabel(
            self.nutrition_frame, 
            text="Nutritional Information", 
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#1E90FF"
        )
        self.nutrition_title.pack(padx=15, pady=(15, 10))
        
        self.nutrition_content = ctk.CTkFrame(self.nutrition_frame, fg_color="transparent")
        self.nutrition_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.placeholder_nutrition = ctk.CTkLabel(
            self.nutrition_content,
            text="Upload an image to see nutritional values",
            font=ctk.CTkFont(size=14),
            text_color="#666666"
        )
        self.placeholder_nutrition.pack(pady=10)
        
        # Status bar
        self.status_bar = ctk.CTkFrame(self.root, height=30, fg_color="#333333")
        self.status_bar.grid(row=1, column=0, sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Status: Ready",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA"
        )
        self.status_label.pack(side=tk.LEFT, padx=15)
        
        self.reset_button = ctk.CTkButton(
            self.status_bar,
            text="Reset",
            command=self.reset_ui,
            fg_color="#444444",
            hover_color="#555555",
            height=22,
            width=80
        )
        self.reset_button.pack(side=tk.RIGHT, padx=15, pady=4)
        
    def update_status(self, message):
        """Update the status bar message"""
        self.status_label.configure(text=f"Status: {message}")
        self.root.update_idletasks()
        
    def load_models(self):
        """Load the classification models"""
        try:
            # In a real application, you would load both models
            # For this example, we'll simulate the model loading
            self.update_status("Loading models...")
            
            # Check if freshness model exists
            if os.path.exists(FRESHNESS_MODEL_PATH):
                self.freshness_model = load_model(FRESHNESS_MODEL_PATH)
                self.update_status("Freshness model loaded")
            else:
                # Create a placeholder model for demo purposes
                self.freshness_model = "dummy_model"
                self.update_status("Using simulated freshness model")
            
            # For food type model, we'll simulate it
            self.food_type_model = "dummy_model"
            
            self.update_status("Ready")
            
        except Exception as e:
            self.update_status(f"Error loading models: {str(e)}")
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        
        if file_path:
            self.current_image_path = file_path
            self.load_and_display_image(file_path)
            self.predict_image()
    
    def load_and_display_image(self, file_path):
        """Load and display the image"""
        try:
            self.update_status(f"Loading image: {Path(file_path).name}")
            
            # Load image
            pil_image = Image.open(file_path)
            
            # Calculate display size while maintaining aspect ratio
            max_width = 750
            max_height = 500
            ratio = min(max_width/pil_image.width, max_height/pil_image.height)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            
            display_image = pil_image.resize(new_size, Image.LANCZOS)
            self.current_image = ImageTk.PhotoImage(display_image)
            
            # Hide placeholder and show image
            self.upload_placeholder.place_forget()
            self.image_label.configure(image=self.current_image)
            self.image_label.place(relx=0.5, rely=0.5, anchor="center")
            
        except Exception as e:
            self.update_status(f"Error loading image: {str(e)}")
    
    def predict_image(self):
        """Run model prediction on the uploaded image"""
        if not self.current_image_path:
            self.update_status("No image selected")
            return
        
        try:
            self.update_status("Analyzing image...")
            
            # Simulate processing steps
            for i in range(5):
                self.update_status(f"Processing... {(i+1)*20}%")
                time.sleep(0.1)
            
            # Since we don't have a real model, let's simulate predictions
            # In a real implementation, you would use your models to predict
            
            # Simulate food type detection (randomly select one for demo)
            food_type = np.random.choice(self.food_types)
            
            # Simulate freshness classification
            is_fresh = np.random.choice([True, False], p=[0.6, 0.4])  # 60% chance of fresh
            confidence = np.random.uniform(0.65, 0.98)
            confidence_percent = confidence * 100
            
            # Update UI with results
            self.food_type_label.configure(text=f"Food Type: {food_type}")
            
            quality_text = "FRESH" if is_fresh else "ROTTEN"
            quality_color = "#2ecc71" if is_fresh else "#e74c3c"
            
            self.freshness_label.configure(
                text=f"Quality: {quality_text} ({confidence_percent:.1f}% confidence)",
                text_color=quality_color
            )
            
            # Update the gauge
            self.draw_gauge(confidence_percent, quality_text)
            
            # Display nutritional information
            self.display_nutrition(food_type)
            
            self.update_status("Analysis complete")
            
        except Exception as e:
            self.update_status(f"Error during analysis: {str(e)}")
    
    def display_nutrition(self, food_type):
        """Display nutritional information for the detected food type"""
        # Clear previous nutrition info
        for widget in self.nutrition_content.winfo_children():
            widget.destroy()
        
        # Get nutritional data for the food type
        nutrition_data = NUTRITIONAL_VALUES.get(food_type, {})
        
        if not nutrition_data:
            label = ctk.CTkLabel(
                self.nutrition_content,
                text="Nutritional information not available",
                font=ctk.CTkFont(size=14),
                text_color="#666666"
            )
            label.pack(pady=10)
            return
        
        # Create a grid layout for nutrition facts
        column_count = 2
        row_count = (len(nutrition_data) + column_count - 1) // column_count
        
        for i, (nutrient, value) in enumerate(nutrition_data.items()):
            row = i // column_count
            col = i % column_count
            
            # Container for each nutrient
            nutrient_frame = ctk.CTkFrame(
                self.nutrition_content, 
                fg_color="#262626", 
                corner_radius=6,
                height=50,
                width=120,
            )
            nutrient_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            nutrient_frame.grid_propagate(False)
            
            # Nutrient name
            name_label = ctk.CTkLabel(
                nutrient_frame,
                text=nutrient,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#AAAAAA"
            )
            name_label.place(relx=0.5, rely=0.3, anchor="center")
            
            # Nutrient value
            value_label = ctk.CTkLabel(
                nutrient_frame,
                text=value,
                font=ctk.CTkFont(size=14),
                text_color="#1E90FF"
            )
            value_label.place(relx=0.5, rely=0.7, anchor="center")
        
        # Configure grid columns to be of equal width
        for i in range(column_count):
            self.nutrition_content.grid_columnconfigure(i, weight=1)
    
    def draw_gauge(self, percentage, label):
        """Draw a circular gauge showing the prediction confidence"""
        self.ax.clear()
        
        # Set colors based on result
        if label == "FRESH":
            color = '#2ecc71'  # Green
        elif label == "ROTTEN":
            color = '#e74c3c'  # Red
        else:
            color = '#bdc3c7'  # Gray
        
        # Draw arc
        self.ax.pie(
            [percentage, 100-percentage],
            startangle=90,
            colors=[color, '#333333'],
            wedgeprops=dict(width=0.3)
        )
        
        # Add percentage text
        self.ax.text(0, 0, f"{percentage:.1f}%", 
                    ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        
        # Add empty circle in the middle
        circle = plt.Circle((0, 0), 0.7, fc='#1A1A1A')
        self.ax.add_artist(circle)
        
        # Add title
        self.ax.set_title(f"{label}", fontsize=14, pad=10, color='white')
        
        # Final adjustments
        self.ax.set_aspect('equal')
        self.ax.set_frame_on(False)
        self.ax.axis('off')
        
        # Update the canvas
        self.canvas.draw()
    
    def reset_ui(self):
        """Reset the UI to initial state"""
        try:
            # Clear image
            self.current_image_path = None
            self.current_image = None
            self.image_label.configure(image=None)
            
            # Show upload placeholder
            self.upload_placeholder.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.8)
            
            # Reset labels
            self.food_type_label.configure(text="Food Type: Awaiting Image")
            self.freshness_label.configure(text="Quality: Awaiting Analysis", text_color="white")
            
            # Reset gauge
            self.draw_gauge(0, "Awaiting Analysis")
            
            # Reset nutrition info
            for widget in self.nutrition_content.winfo_children():
                widget.destroy()
                
            self.placeholder_nutrition = ctk.CTkLabel(
                self.nutrition_content,
                text="Upload an image to see nutritional values",
                font=ctk.CTkFont(size=14),
                text_color="#666666"
            )
            self.placeholder_nutrition.pack(pady=10)
            
            # Update status
            self.update_status("Reset complete")
            
        except Exception as e:
            self.update_status(f"Error during reset: {str(e)}")

def main():
    root = ctk.CTk()
    app = FoodClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()