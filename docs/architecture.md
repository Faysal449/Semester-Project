# System Architecture

The system is composed of two main components.

Windows PC
-----------
• Runs CARLA simulator
• Spawns vehicle and sensors
• Extracts camera frames
• Sends sensor data to Jetson Nano via TCP

Jetson Nano
-----------
• Receives frames
• Processes images
• (future) runs AI inference
• (future) sends control commands

Data Flow
----------
CARLA Camera → Frame Preprocessing → TCP Transmission →
Jetson Nano → Image Processing

CARLA (Windows)
      │
      │ Camera Frames
      ▼
TCP Network
      │
      ▼
Jetson Nano
      │
      ▼
Image Processing / AI
