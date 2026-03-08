# Communication Protocol

Transport
---------
TCP socket communication

Default Port
------------
5000

Frame Structure
---------------

Header:
• frame_size (4 bytes)

Payload:
• grayscale image bytes

Example Flow
------------

CARLA Camera Frame
      ↓
Grayscale Conversion
      ↓
Serialized Frame
      ↓
TCP Transmission
      ↓
Jetson Receiver
