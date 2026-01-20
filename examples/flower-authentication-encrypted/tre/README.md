# TRE Flower Federated Learning Microservices

This directory contains Flask microservices that expose Flower's federated learning capabilities through REST APIs, designed for integration with Trusted Research Environment (TRE) systems.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CENTRAL ANALYZER SIDE                            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │               Analyzer Service (Flask REST API)                  │   │
│  │                     http://localhost:5000                        │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │                    SuperLink Process                     │    │   │
│  │  │  Fleet API: 9092 | Control API: 9093 | ServerApp: 9091  │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                      TLS + Authentication
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       │                            │                            │
       ▼                            ▼                            ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  DATA OWNER SITE 1  │  │  DATA OWNER SITE 2  │  │  DATA OWNER SITE N  │
│                     │  │                     │  │                     │
│ ┌─────────────────┐ │  │ ┌─────────────────┐ │  │ ┌─────────────────┐ │
│ │ DataOwner Svc   │ │  │ │ DataOwner Svc   │ │  │ │ DataOwner Svc   │ │
│ │ REST API: 5001  │ │  │ │ REST API: 5002  │ │  │ │ REST API: 500N  │ │
│ │ ┌─────────────┐ │ │  │ │ ┌─────────────┐ │ │  │ │ ┌─────────────┐ │ │
│ │ │  SuperNode  │ │ │  │ │ │  SuperNode  │ │ │  │ │ │  SuperNode  │ │ │
│ │ └─────────────┘ │ │  │ │ └─────────────┘ │ │  │ │ └─────────────┘ │ │
│ └─────────────────┘ │  │ └─────────────────┘ │  │ └─────────────────┘ │
│ [Local Dataset]     │  │ [Local Dataset]     │  │ [Local Dataset]     │
│ [Encryption Keys]   │  │ [Encryption Keys]   │  │ [Encryption Keys]   │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

## Components

### Analyzer Service (`analyzer_service/`)

The central coordinator service that:
- Manages the Flower SuperLink server lifecycle
- Handles SuperNode registration/unregistration
- Orchestrates federated learning runs
- Manages TLS certificates and authentication keys

**REST API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Overall service status |
| POST | `/superlink/start` | Start SuperLink server |
| POST | `/superlink/stop` | Stop SuperLink server |
| GET | `/superlink/status` | Get SuperLink status |
| GET | `/superlink/logs` | Get SuperLink logs |
| GET | `/supernodes` | List registered SuperNodes |
| POST | `/supernodes/register` | Register a new SuperNode |
| POST | `/supernodes/<id>/unregister` | Unregister a SuperNode |
| GET | `/runs` | List all FL runs |
| POST | `/runs` | Start a new FL run |
| GET | `/runs/<id>` | Get run details |
| POST | `/runs/<id>/stop` | Stop a running FL run |
| GET | `/runs/<id>/logs` | Get run logs |
| POST | `/certificates/generate` | Generate TLS certificates |
| GET | `/certificates` | List certificates |
| POST | `/keys/generate` | Generate node keys |
| GET | `/keys` | List available keys |

### Data Owner Service (`dataowner_service/`)

The client-side service deployed at each data owner site that:
- Manages the Flower SuperNode client lifecycle
- Handles local dataset management
- Manages node-specific keys and certificates
- Monitors training progress

**REST API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Overall node status |
| POST | `/supernode/start` | Start SuperNode client |
| POST | `/supernode/stop` | Stop SuperNode client |
| GET | `/supernode/status` | Get SuperNode status |
| GET | `/supernode/logs` | Get SuperNode logs |
| GET | `/dataset` | Get dataset info |
| POST | `/dataset/upload` | Upload dataset file |
| GET | `/dataset/download` | Download dataset |
| GET | `/keys` | Get keys info |
| POST | `/keys/upload/auth` | Upload auth keys |
| POST | `/keys/upload/encryption` | Upload encryption key |
| GET | `/keys/download/public` | Download public key |
| GET | `/certificates` | Get certificates info |
| POST | `/certificates/upload/ca` | Upload CA certificate |
| GET | `/models` | List model artifacts |
| GET | `/models/<filename>` | Download model |
| GET | `/config` | Get node configuration |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- OpenSSL (for certificate generation)

### 1. Setup and Deploy

```bash
cd tre

# Full setup (generates certs, keys, datasets, builds, and starts)
./build_and_deploy.sh all

# Or step by step:
./build_and_deploy.sh setup    # Generate credentials and datasets
./build_and_deploy.sh build    # Build Docker images
./build_and_deploy.sh start    # Start services
```

### 2. Check Service Status

```bash
# Check all services
./build_and_deploy.sh status

# Health check
curl http://localhost:5000/health  # Analyzer
curl http://localhost:5001/health  # Data Owner 1
curl http://localhost:5002/health  # Data Owner 2
```

### 3. Run Federated Learning

```bash
# Using the orchestration script
./build_and_deploy.sh run

# Or manually with the Python script
python3 scripts/orchestrate.py --num-rounds 3 --local-epochs 1
```

### 4. Stop Services

```bash
./build_and_deploy.sh stop
```

## Manual Workflow via REST APIs

### Step 1: Start SuperLink on Analyzer

```bash
curl -X POST http://localhost:5000/superlink/start \
  -H "Content-Type: application/json" \
  -d '{
    "insecure": false,
    "enable_supernode_auth": true
  }'
```

### Step 2: Register SuperNodes

```bash
# Register node 1
curl -X POST http://localhost:5000/supernodes/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "1",
    "generate_keys": false,
    "public_key_path": "/app/data/keys/client_credentials_1.pub"
  }'

# Register node 2
curl -X POST http://localhost:5000/supernodes/register \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "2",
    "public_key_path": "/app/data/keys/client_credentials_2.pub"
  }'
```

### Step 3: Start SuperNodes on Data Owners

```bash
# Start SuperNode on Data Owner 1
curl -X POST http://localhost:5001/supernode/start \
  -H "Content-Type: application/json" \
  -d '{
    "superlink_address": "analyzer:9092"
  }'

# Start SuperNode on Data Owner 2
curl -X POST http://localhost:5002/supernode/start \
  -H "Content-Type: application/json" \
  -d '{
    "superlink_address": "analyzer:9092"
  }'
```

### Step 4: Start Federated Learning Run

```bash
curl -X POST http://localhost:5000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "run_config": {
      "num-server-rounds": 3,
      "local-epochs": 1
    }
  }'
```

### Step 5: Monitor Progress

```bash
# List runs
curl http://localhost:5000/runs

# Get SuperLink logs
curl http://localhost:5000/superlink/logs?lines=50

# Get SuperNode logs
curl http://localhost:5001/supernode/logs?lines=50
```

## Directory Structure

```
tre/
├── analyzer_service/          # Analyzer (SuperLink) Flask service
│   ├── __init__.py
│   └── app.py                 # Main Flask application
├── dataowner_service/         # Data Owner (SuperNode) Flask service
│   ├── __init__.py
│   └── app.py                 # Main Flask application
├── shared/                    # Shared utilities
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── process_manager.py     # Subprocess management
│   ├── cli_wrapper.py         # Flower CLI wrapper
│   └── key_generator.py       # Key generation utilities
├── scripts/                   # Orchestration scripts
│   ├── __init__.py
│   ├── setup.py               # Environment setup
│   └── orchestrate.py         # Workflow orchestration
├── data/                      # Data directory (created by setup)
│   ├── certificates/          # TLS certificates
│   ├── keys/                  # Authentication keys
│   ├── encryption_keys/       # Encryption keys
│   └── datasets/              # Dataset partitions
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile.analyzer        # Analyzer service Dockerfile
├── Dockerfile.dataowner       # Data Owner service Dockerfile
├── requirements.txt           # Python dependencies
├── build_and_deploy.sh        # Build and deployment script
└── README.md                  # This file
```

## Configuration

### Environment Variables

**Analyzer Service:**

| Variable | Default | Description |
|----------|---------|-------------|
| `FLOWER_APP_PATH` | `/app/flower-app` | Path to Flower app |
| `DATA_DIR` | `/app/data` | Base data directory |
| `CERTIFICATES_DIR` | `/app/data/certificates` | TLS certificates |
| `KEYS_DIR` | `/app/data/keys` | Authentication keys |
| `ENCRYPTION_KEYS_DIR` | `/app/data/encryption_keys` | Encryption keys |

**Data Owner Service:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ID` | `1` | Unique node identifier |
| `DATA_DIR` | `/app/data` | Base data directory |
| `DATASET_PATH` | `/app/data/datasets/cifar10_part_1` | Dataset location |
| `SUPERLINK_ADDRESS` | `superlink:9092` | SuperLink Fleet API |

## Security Features

### TLS/SSL

All communications between SuperLink and SuperNodes are encrypted using TLS:
- CA certificate validates server identity
- Server certificate authenticates the SuperLink
- Mutual TLS can be enabled for additional security

### SuperNode Authentication

Each SuperNode has a unique ECDSA key pair:
- Public key is registered with the SuperLink
- Private key is used by the SuperNode to authenticate

### Encrypted Weight Transmission

Model weights are encrypted using AES-256-GCM:
- Each node has a unique 256-bit encryption key
- Weights are encrypted before transmission
- Server decrypts using node-specific keys

## Extending for Production

### Key Management

Replace the file-based key storage with:
- HashiCorp Vault
- AWS KMS
- Azure Key Vault
- Hardware Security Modules (HSM)

### Database Backend

The PostgreSQL container is included for future use:
- Store run history and metrics
- Manage node registrations
- Track model versions

### Monitoring

Add observability with:
- Prometheus metrics endpoint
- Grafana dashboards
- Distributed tracing

### High Availability

For production deployments:
- Multiple SuperLink replicas with load balancing
- Redis for session/state sharing
- Kubernetes deployment with health checks

## Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker logs
docker-compose logs analyzer
docker-compose logs dataowner_1
```

**Connection refused:**
```bash
# Verify services are running
docker-compose ps

# Check network connectivity
docker exec tre_dataowner_1 curl http://analyzer:5000/health
```

**Certificate errors:**
```bash
# Regenerate certificates
rm -rf data/certificates
./build_and_deploy.sh setup
```

**SuperNode registration fails:**
```bash
# Ensure SuperLink is running
curl http://localhost:5000/superlink/status

# Check keys exist
ls -la data/keys/
```

## License

Apache 2.0 - See the main Flower repository for details.
