#!/usr/bin/env bash
set -e

echo "Installing ComfyUI Manager..."
pip install -U --pre comfyui-manager

echo ""
echo "ComfyUI Manager installed successfully."
echo ""
echo "Restart ComfyUI with the following command to enable the manager:"
echo ""
echo "  python main.py --enable-manager"
echo ""
echo "After restarting, open the ComfyUI web interface and use"
echo "Manager > Install Missing Custom Nodes to install all required nodes."
