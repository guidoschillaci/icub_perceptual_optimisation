echo "pulling most recent version"
git pull
echo "Clearing previous results"
sh scripts/clear_results.sh
echo "Running main script"
python3 scripts/main.py
echo "Run script done!"