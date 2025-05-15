echo "Starting uninstallation of StockAI python packages..."
pip uninstall -r requirements.txt -y
sudo rm -rf Stock-AI/
echo "Uninstallation of python packages completed."
