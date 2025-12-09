#!/bin/bash

# Generate encryption keys for SuperNodes
# Usage: ./generate_encryption_keys.sh [number_of_nodes]
# Default: 2 nodes

set -e

NUM_NODES=${1:-2}

echo "Generating encryption keys for $NUM_NODES nodes..."

# Create encryption_keys directory
mkdir -p encryption_keys

# Generate encryption keys for each node
for ((i=1; i<=NUM_NODES; i++)); do
    KEY_FILE="encryption_keys/node-${i}_key.bin"

    # Generate a random 32-byte key using Python
    python3 << EOF
import os
with open("$KEY_FILE", "wb") as f:
    f.write(os.urandom(32))
print(f"Generated encryption key: $KEY_FILE")
EOF
done

echo "Encryption key generation complete!"
echo "Keys stored in encryption_keys/ directory"
