#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Please specify path to python3 executable: ./init.sh <path-to-python3-binary>"
    exit 1
fi;

python3_path="$1"

"$python3_path" -m pip install --upgrade virtualenv
"$python3_path" -m virtualenv -p "$python3_path" venv
source venv/bin/activate
python -m pip install -r requirements.txt

cat << EOF

To activate virtual enviroment use:
source venv/bin/activate
EOF

