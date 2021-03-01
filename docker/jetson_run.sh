echo "pulling most recent version"
git pull
echo "Clearing previous results"
sh docker/jetson_clear_results.sh
echo "Running main script"
python3 scripts/main.py
echo "Run script done!"