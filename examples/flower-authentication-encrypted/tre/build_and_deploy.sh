#!/bin/bash
# Build and deploy TRE Flower microservices
#
# This script:
# 1. Generates certificates, keys, and datasets
# 2. Builds Docker images
# 3. Starts the services
# 4. Optionally runs the federated learning workflow

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
NUM_NODES=${NUM_NODES:-2}
DATA_DIR="${SCRIPT_DIR}/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Setup data directory and generate credentials
setup_data() {
    log_info "Setting up data directory..."

    # Create directories
    mkdir -p "$DATA_DIR/certificates"
    mkdir -p "$DATA_DIR/keys"
    mkdir -p "$DATA_DIR/encryption_keys"

    for i in $(seq 1 $NUM_NODES); do
        mkdir -p "$DATA_DIR/datasets/cifar10_part_$i"
    done

    log_info "Running setup script..."

    # Check if setup.py exists
    if [ -f "scripts/setup.py" ]; then
        cd "$SCRIPT_DIR"
        python3 scripts/setup.py --base-dir "$DATA_DIR" --num-nodes "$NUM_NODES"
    else
        log_warn "Setup script not found, generating credentials manually..."

        # Generate certificates
        if [ ! -f "$DATA_DIR/certificates/ca.crt" ]; then
            log_info "Generating TLS certificates..."

            # Create certificate config
            cat > "$DATA_DIR/certificates/certificate.conf" << EOF
[req]
default_bits = 4096
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn

[dn]
C = US
ST = CA
O = TRE Server
CN = localhost

[req_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = analyzer
DNS.3 = superlink
DNS.4 = dataowner_1
DNS.5 = dataowner_2
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
EOF

            # Generate CA
            openssl genrsa -out "$DATA_DIR/certificates/ca.key" 4096
            openssl req -new -x509 -key "$DATA_DIR/certificates/ca.key" -sha256 \
                -subj "/C=US/ST=CA/O=TRE CA, Inc." -days 365 \
                -out "$DATA_DIR/certificates/ca.crt"

            # Generate server cert
            openssl genrsa -out "$DATA_DIR/certificates/server.key" 4096
            openssl req -new -key "$DATA_DIR/certificates/server.key" \
                -out "$DATA_DIR/certificates/server.csr" \
                -config "$DATA_DIR/certificates/certificate.conf"
            openssl x509 -req -in "$DATA_DIR/certificates/server.csr" \
                -CA "$DATA_DIR/certificates/ca.crt" -CAkey "$DATA_DIR/certificates/ca.key" \
                -CAcreateserial -out "$DATA_DIR/certificates/server.pem" -days 365 -sha256 \
                -extfile "$DATA_DIR/certificates/certificate.conf" -extensions req_ext
            rm "$DATA_DIR/certificates/server.csr"
        fi

        # Generate SSH keys for authentication
        for i in $(seq 1 $NUM_NODES); do
            if [ ! -f "$DATA_DIR/keys/client_credentials_$i" ]; then
                log_info "Generating authentication keys for node $i..."
                ssh-keygen -t ecdsa -b 384 -N "" -f "$DATA_DIR/keys/client_credentials_$i" -C ""
            fi
        done

        # Generate encryption keys
        for i in $(seq 1 $NUM_NODES); do
            if [ ! -f "$DATA_DIR/encryption_keys/node-${i}_key.bin" ]; then
                log_info "Generating encryption key for node $i..."
                python3 -c "import os; open('$DATA_DIR/encryption_keys/node-${i}_key.bin', 'wb').write(os.urandom(32))"
            fi
        done
    fi

    log_info "Data setup complete"
}

# Prepare datasets
prepare_datasets() {
    log_info "Preparing datasets..."

    # Go to parent example directory
    EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"

    if [ -f "$EXAMPLE_DIR/prepare_dataset_simple.py" ]; then
        cd "$EXAMPLE_DIR"
        python3 prepare_dataset_simple.py

        # Copy datasets to tre/data
        for i in $(seq 1 $NUM_NODES); do
            if [ -d "$EXAMPLE_DIR/datasets/cifar10_part_$i" ] && [ ! -d "$DATA_DIR/datasets/cifar10_part_$i/data.pt" ]; then
                log_info "Copying dataset partition $i..."
                cp -r "$EXAMPLE_DIR/datasets/cifar10_part_$i"/* "$DATA_DIR/datasets/cifar10_part_$i/" 2>/dev/null || true
            fi
        done
    else
        log_warn "Dataset preparation script not found"
        log_warn "You may need to prepare datasets manually"
    fi

    cd "$SCRIPT_DIR"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."

    cd "$SCRIPT_DIR"

    # Use docker compose v2 syntax if available
    if docker compose version &> /dev/null; then
        docker compose build
    else
        docker-compose build
    fi

    log_info "Docker images built successfully"
}

# Start services
start_services() {
    log_info "Starting services..."

    cd "$SCRIPT_DIR"

    if docker compose version &> /dev/null; then
        docker compose up -d
    else
        docker-compose up -d
    fi

    log_info "Services started"
    log_info "Waiting for services to be healthy..."

    # Wait for services
    sleep 10

    # Check health
    for service in analyzer dataowner_1 dataowner_2; do
        port=5000
        case $service in
            analyzer) port=5000 ;;
            dataowner_1) port=5001 ;;
            dataowner_2) port=5002 ;;
        esac

        if curl -s "http://localhost:$port/health" | grep -q "success"; then
            log_info "$service is healthy"
        else
            log_warn "$service may not be ready yet"
        fi
    done
}

# Stop services
stop_services() {
    log_info "Stopping services..."

    cd "$SCRIPT_DIR"

    if docker compose version &> /dev/null; then
        docker compose down
    else
        docker-compose down
    fi

    log_info "Services stopped"
}

# Show logs
show_logs() {
    cd "$SCRIPT_DIR"

    if docker compose version &> /dev/null; then
        docker compose logs -f
    else
        docker-compose logs -f
    fi
}

# Run orchestration
run_workflow() {
    log_info "Running federated learning workflow..."

    cd "$SCRIPT_DIR"

    if [ -f "scripts/orchestrate.py" ]; then
        python3 scripts/orchestrate.py "$@"
    else
        log_error "Orchestration script not found"
        exit 1
    fi
}

# Print status
print_status() {
    log_info "Service Status:"
    echo ""
    echo "Analyzer Service:"
    echo "  REST API:    http://localhost:5000"
    echo "  Fleet API:   localhost:9092"
    echo "  Control API: localhost:9093"
    echo ""
    echo "Data Owner 1:"
    echo "  REST API:    http://localhost:5001"
    echo ""
    echo "Data Owner 2:"
    echo "  REST API:    http://localhost:5002"
    echo ""
    echo "Useful commands:"
    echo "  Check health:   curl http://localhost:5000/health"
    echo "  View logs:      $0 logs"
    echo "  Run workflow:   $0 run"
    echo "  Stop services:  $0 stop"
}

# Print usage
usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup       Setup data directory and generate credentials"
    echo "  build       Build Docker images"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  logs        Show service logs"
    echo "  status      Show service status and URLs"
    echo "  run         Run federated learning workflow"
    echo "  all         Setup, build, start, and run (default)"
    echo "  clean       Stop services and remove data"
    echo ""
    echo "Environment variables:"
    echo "  NUM_NODES   Number of data owner nodes (default: 2)"
}

# Main
case "${1:-all}" in
    setup)
        check_prerequisites
        setup_data
        prepare_datasets
        ;;
    build)
        check_prerequisites
        build_images
        ;;
    start)
        check_prerequisites
        start_services
        print_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        print_status
        ;;
    logs)
        show_logs
        ;;
    status)
        print_status
        ;;
    run)
        shift
        run_workflow "$@"
        ;;
    all)
        check_prerequisites
        setup_data
        prepare_datasets
        build_images
        start_services
        print_status
        log_info ""
        log_info "To run federated learning, execute:"
        log_info "  $0 run"
        ;;
    clean)
        stop_services
        log_info "Removing data directory..."
        rm -rf "$DATA_DIR"
        log_info "Clean complete"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
