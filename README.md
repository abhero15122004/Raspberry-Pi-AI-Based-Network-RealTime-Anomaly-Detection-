# Raspberry-Pi-AI-Based-Network-RealTime-Anomaly-Detection-
A low cost and scalable project designed using raspberry pi to find real time anomalies in a wifi network between multiple devices connected to the network
# Raspberry Pi Network Anomaly Detection System

This project implements a real-time network anomaly detection system using Raspberry Pi 4 and unsupervised machine learning (Isolation Forest + PCA).

## Features
- Captures network packets using tcpdump/tshark
- Processes packet data into CSV
- Runs anomaly detection using hybrid ML models
- Saves detected anomalies to CSV

## Requirements
- Raspberry Pi 4 (4GB RAM)
- USB-to-Ethernet Adapter
- Python 3.x

## Usage
1. Capture traffic:
   ```bash
   sudo tshark -i eth1 -T fields -e frame.time_epoch -e ip.src -e ip.dst -e ip.len -e ip.proto > network_data.csv

   Train models:
   python3 train_model.py

   Detect anomalies:
   python3 detect_anomalies.py

   
